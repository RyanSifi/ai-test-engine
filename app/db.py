import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values
from contextlib import contextmanager
from typing import List, Dict, Optional


# ID arbitraire et stable pour pg_advisory_lock — empêche la course
# entre plusieurs workers uvicorn qui appellent init_schema simultanément.
_INIT_LOCK_ID = 8472_31415


class KnowledgeDB:
    """
    Gère toutes les interactions avec PostgreSQL/pgvector :
    schéma, indexation des chunks de code et recherche sémantique.

    Utilise un pool de connexions thread-safe (psycopg2.pool.ThreadedConnectionPool).
    """

    def __init__(self, db_url: str, minconn: int = 1, maxconn: int = 10):
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

    def init_schema(self, vector_size: int = 384) -> None:
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

    def reset_schema(self, vector_size: int = 384) -> None:
        """
        Supprime et recrée la table principale.
        À utiliser uniquement lors d'un changement de modèle d'embedding.
        Attention toutes les données sont perdues.
        """
        with self._cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS routes CASCADE;")
            cur.execute("DROP TABLE IF EXISTS scenarios CASCADE;")
            cur.execute("DROP TABLE IF EXISTS project_code_context CASCADE;")
            self._create_tables(cur, vector_size)

    def _create_tables(self, cur, vector_size: int) -> None:
        """Crée l'extension pgvector et la table de chunks de code."""
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

    def save_code_context(
        self,
        project_id: str,
        chunks: List[Dict],
        vectors: List[List[float]],
    ) -> None:
        """Insère les chunks et leurs embeddings. Atomique (commit unique)."""
        with self._cursor() as cur:
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
        limit: int = 8,
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
