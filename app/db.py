import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values
from contextlib import contextmanager
from typing import List, Dict, Optional


# ID arbitraire et stable pour pg_advisory_lock — empêche la course
# entre plusieurs workers uvicorn qui appellent init_schema simultanément.
_INIT_LOCK_ID = 8472_31415

# ── Dimensions de vecteur acceptées ──────────────────────────────────────────
# La taille d'un vecteur pgvector fait partie du TYPE de la colonne : elle ne
# peut donc pas être passée en paramètre lié (`%s`), seule l'interpolation est
# possible dans le CREATE TABLE. C'est le seul endroit du projet où du SQL est
# construit par interpolation — d'où la validation stricte ci-dessous.
#
# En pratique la valeur vient toujours de `len(embedding)`, donc d'un entier
# calculé par le modèle, jamais d'une entrée utilisateur. La validation relève
# de la défense en profondeur : elle garantit que même un appel interne fautif
# ne peut pas injecter de SQL, plutôt que de reposer sur cette seule promesse.
MIN_VECTOR_SIZE = 1
MAX_VECTOR_SIZE = 16000   # limite dure de pgvector pour un index HNSW
DEFAULT_VECTOR_SIZE = 384  # dimension de paraphrase-multilingual-MiniLM-L12-v2

# Pool de connexions. maxconn borne le nombre de requêtes concurrentes vers
# PostgreSQL : au-delà, les appels attendent au lieu d'ouvrir des connexions à
# l'infini. 10 couvre largement l'usage (un utilisateur, quelques générations en
# tâche de fond) sans saturer le `max_connections` par défaut de PostgreSQL (100).
DEFAULT_MIN_CONNECTIONS = 1
DEFAULT_MAX_CONNECTIONS = 10

# Nombre de voisins remontés par défaut lors d'une recherche vectorielle.
# Les appelants passent une valeur explicite (cf. rag_context.RAG_LIMIT_*) ;
# celle-ci ne sert que de garde-fou si l'argument est omis.
DEFAULT_SEARCH_LIMIT = 8


def _validate_vector_size(vector_size) -> int:
    """
    Valide la dimension avant toute interpolation dans du SQL.

    Refuse tout ce qui n'est pas un entier dans les bornes de pgvector — y
    compris les booléens, que Python considère comme des entiers.
    """
    if isinstance(vector_size, bool) or not isinstance(vector_size, int):
        raise ValueError(
            f"vector_size doit être un entier, reçu {type(vector_size).__name__} "
            f"({vector_size!r})"
        )
    if not (MIN_VECTOR_SIZE <= vector_size <= MAX_VECTOR_SIZE):
        raise ValueError(
            f"vector_size hors bornes : {vector_size} "
            f"(attendu entre {MIN_VECTOR_SIZE} et {MAX_VECTOR_SIZE})"
        )
    return vector_size


class KnowledgeDB:
    """
    Gère toutes les interactions avec PostgreSQL/pgvector :
    schéma, indexation des chunks de code et recherche sémantique.

    Utilise un pool de connexions thread-safe (psycopg2.pool.ThreadedConnectionPool).
    """

    def __init__(self, db_url: str,
                 minconn: int = DEFAULT_MIN_CONNECTIONS,
                 maxconn: int = DEFAULT_MAX_CONNECTIONS):
        self.db_url = db_url
        self._pool: Optional[pool.ThreadedConnectionPool] = None
        if db_url:
            self._pool = pool.ThreadedConnectionPool(minconn, maxconn, db_url)

    # ------------------------------------------------------------------
    # Connexion / context manager
    # ------------------------------------------------------------------

    @contextmanager
    def _cursor(self, commit: bool = True):
        """
        Acquiert une connexion du pool, ouvre un curseur, et la rend au pool
        à la fin (rollback si exception, commit sinon par défaut).
        """
        if self._pool is None:
            raise RuntimeError("Pool de connexions non initialisé.")
        conn = self._pool.getconn()
        try:
            cur = conn.cursor()
            try:
                yield cur
                if commit:
                    conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cur.close()
        finally:
            self._pool.putconn(conn)

    # ------------------------------------------------------------------
    # Schéma
    # ------------------------------------------------------------------

    def init_schema(self, vector_size: int = DEFAULT_VECTOR_SIZE) -> None:
        """
        Crée la table project_code_context si elle n'existe pas encore.
        Sérialisé via pg_advisory_lock pour éviter les courses entre workers.
        """
        with self._cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(%s);", (_INIT_LOCK_ID,))
            try:
                cur.execute("SELECT to_regclass('public.project_code_context');")
                if cur.fetchone()[0] is not None:
                    return
                self._create_tables(cur, vector_size)
            finally:
                cur.execute("SELECT pg_advisory_unlock(%s);", (_INIT_LOCK_ID,))

    def reset_schema(self, vector_size: int = DEFAULT_VECTOR_SIZE) -> None:
        """
        Supprime et recrée la table principale.
        À utiliser uniquement lors d'un changement de modèle d'embedding.
        Attention toutes les données sont perdues.
        """
        with self._cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS project_code_context CASCADE;")
            self._create_tables(cur, vector_size)

    def _create_tables(self, cur, vector_size: int) -> None:
        """
        Crée l'extension pgvector et la table de chunks de code.

        `vector_size` est interpolé dans le SQL — c'est inévitable, la dimension
        fait partie du type de colonne et n'accepte pas de paramètre lié. La
        valeur est donc validée en amont par _validate_vector_size(), qui refuse
        tout ce qui n'est pas un entier dans les bornes de pgvector. Toutes les
        autres requêtes du projet utilisent des paramètres liés (%s).
        """
        vector_size = _validate_vector_size(vector_size)
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        cur.execute(f"""
            CREATE TABLE project_code_context (
                id          SERIAL PRIMARY KEY,
                project_id  VARCHAR(100),
                chunk_type  VARCHAR(50),
                file_path   VARCHAR(500),
                class_name  VARCHAR(255),
                content     TEXT,
                embedding   vector({vector_size})
            );
        """)

        cur.execute("CREATE INDEX ON project_code_context USING hnsw (embedding vector_cosine_ops);")
        cur.execute("CREATE INDEX ON project_code_context (project_id);")

    # ------------------------------------------------------------------
    # CRUD – projets
    # ------------------------------------------------------------------

    def clear_project(self, project_id: str) -> None:
        """Supprime toutes les données d'un projet (pour ré-indexation)."""
        with self._cursor() as cur:
            cur.execute("DELETE FROM project_code_context WHERE project_id = %s", (project_id,))

    def list_projects(self) -> List[str]:
        """Retourne la liste de tous les project_id indexés."""
        with self._cursor(commit=False) as cur:
            cur.execute(
                "SELECT DISTINCT project_id FROM project_code_context ORDER BY project_id;"
            )
            return [row[0] for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------

    def reindex_project(
        self,
        project_id: str,
        chunks: List[Dict],
        vectors: List[List[float]],
    ) -> None:
        """
        Supprime puis ré-insère les chunks d'un projet — atomique.
        Évite la fenêtre de temps où le projet apparaît vide entre clear et save.
        """
        with self._cursor() as cur:
            cur.execute("DELETE FROM project_code_context WHERE project_id = %s", (project_id,))
            if chunks:
                data = [
                    (
                        project_id,
                        chunk["chunk_type"],
                        chunk["file_path"],
                        chunk["class_name"],
                        chunk["content"],
                        vector,
                    )
                    for chunk, vector in zip(chunks, vectors)
                ]
                execute_values(
                    cur,
                    """
                    INSERT INTO project_code_context
                        (project_id, chunk_type, file_path, class_name, content, embedding)
                    VALUES %s
                    """,
                    data,
                )

    # ------------------------------------------------------------------
    # Recherche
    # ------------------------------------------------------------------

    def find_closest_code_context(
        self,
        project_id: str,
        query_vector: List[float],
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> List[Dict]:
        """Retourne les N chunks de code les plus proches sémantiquement."""
        with self._cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT chunk_type, file_path, class_name, content,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM   project_code_context
                WHERE  project_id = %s
                ORDER  BY embedding <=> %s::vector
                LIMIT  %s;
                """,
                (query_vector, project_id, query_vector, limit),
            )
            return [
                {
                    "chunk_type":  row[0],
                    "file_path":   row[1],
                    "class_name":  row[2],
                    "content":     row[3],
                    "similarity":  float(row[4]),
                }
                for row in cur.fetchall()
            ]

    def get_code_by_class_name(self, project_id: str, class_name: str) -> List[Dict]:
        """Récupère les chunks correspondant à une classe par son nom (ILIKE)."""
        with self._cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT chunk_type, file_path, class_name, content
                FROM   project_code_context
                WHERE  project_id  = %s
                  AND  class_name ILIKE %s;
                """,
                (project_id, f"%{class_name}%"),
            )
            return [
                {
                    "chunk_type": row[0],
                    "file_path":  row[1],
                    "class_name": row[2],
                    "content":    row[3],
                }
                for row in cur.fetchall()
            ]

    def get_project_stats(self, project_id: str) -> Dict:
        """Retourne le nombre de chunks indexés pour un projet."""
        with self._cursor(commit=False) as cur:
            cur.execute(
                "SELECT COUNT(*) FROM project_code_context WHERE project_id = %s",
                (project_id,),
            )
            chunk_count = cur.fetchone()[0]
            return {
                "project_id":  project_id,
                "chunks":      chunk_count,
            }
