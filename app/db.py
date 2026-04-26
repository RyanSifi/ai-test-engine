import psycopg2
from psycopg2.extras import execute_values
import json
from typing import List, Dict


class KnowledgeDB:
    """
    Gère toutes les interactions avec PostgreSQL/pgvector :
    schéma, indexation des chunks de code et recherche sémantique.
    """

    def __init__(self, db_url: str):
        self.db_url = db_url

    # ------------------------------------------------------------------
    # Connexion
    # ------------------------------------------------------------------

    def get_conn(self):
        """Ouvre et retourne une connexion PostgreSQL."""
        # TODO: remplacer par un connection pool (psycopg2.pool.ThreadedConnectionPool
        # ou SQLAlchemy) pour éviter la saturation sous charge concurrente.
        # Actuellement chaque méthode ouvre et ferme sa propre connexion.
        return psycopg2.connect(self.db_url)

    # ------------------------------------------------------------------
    # Schéma
    # ------------------------------------------------------------------

    def init_schema(self, vector_size: int = 384) -> None:
        """
        Crée les tables si elles n'existent pas encore.
        Ne fait RIEN si les tables sont déjà présentes (évite d'effacer
        des données lors d'un simple redémarrage du service).
        Pour recréer le schéma (ex. changement de modèle d'embedding),
        appeler reset_schema() explicitement.
        """
        conn = self.get_conn()
        cur  = conn.cursor()
        try:
            cur.execute("SELECT to_regclass('public.project_code_context');")
            already_exists = cur.fetchone()[0] is not None
            if already_exists:
                return  # Tables OK, rien à faire
            self._create_tables(cur, vector_size)
            conn.commit()
        finally:
            cur.close()
            conn.close()

    def reset_schema(self, vector_size: int = 384) -> None:
        """
        Supprime et recrée toutes les tables.
        À utiliser uniquement lors d'un changement de modèle d'embedding
        (dimension des vecteurs différente) ou d'une réinitialisation complète.
        ⚠️  Toutes les données sont perdues.
        """
        conn = self.get_conn()
        cur  = conn.cursor()
        try:
            cur.execute("DROP TABLE IF EXISTS routes CASCADE;")
            cur.execute("DROP TABLE IF EXISTS scenarios CASCADE;")
            cur.execute("DROP TABLE IF EXISTS project_code_context CASCADE;")
            self._create_tables(cur, vector_size)
            conn.commit()
        finally:
            cur.close()
            conn.close()

    def _create_tables(self, cur, vector_size: int) -> None:
        """Crée l'extension pgvector et les trois tables du schéma."""
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        cur.execute(f"""
            CREATE TABLE routes (
                id         SERIAL PRIMARY KEY,
                project_id VARCHAR(100),
                path       VARCHAR(500),
                method     VARCHAR(10),
                name       VARCHAR(255),
                embedding  vector({vector_size})
            );
        """)

        cur.execute(f"""
            CREATE TABLE scenarios (
                id          SERIAL PRIMARY KEY,
                project_id  VARCHAR(100),
                description TEXT,
                input_json  TEXT,
                embedding   vector({vector_size})
            );
        """)

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

        cur.execute("CREATE INDEX ON routes               USING hnsw (embedding vector_cosine_ops);")
        cur.execute("CREATE INDEX ON project_code_context USING hnsw (embedding vector_cosine_ops);")

    # ------------------------------------------------------------------
    # CRUD – projets
    # ------------------------------------------------------------------

    def clear_project(self, project_id: str) -> None:
        """Supprime toutes les données d'un projet (pour ré-indexation)."""
        conn = self.get_conn()
        cur  = conn.cursor()
        try:
            cur.execute("DELETE FROM routes                WHERE project_id = %s", (project_id,))
            cur.execute("DELETE FROM scenarios             WHERE project_id = %s", (project_id,))
            cur.execute("DELETE FROM project_code_context WHERE project_id = %s", (project_id,))
            conn.commit()
        finally:
            cur.close()
            conn.close()

    def list_projects(self) -> List[str]:
        """Retourne la liste de tous les project_id indexés."""
        conn = self.get_conn()
        cur  = conn.cursor()
        try:
            cur.execute("""
                SELECT DISTINCT project_id FROM project_code_context
                UNION
                SELECT DISTINCT project_id FROM routes
                ORDER BY project_id;
            """)
            return [row[0] for row in cur.fetchall()]
        finally:
            cur.close()
            conn.close()

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------

    def save_routes(self, project_id: str, routes: List, vectors: List) -> None:
        conn = self.get_conn()
        cur  = conn.cursor()
        try:
            data = [(project_id, r.path, r.method, r.name, v)
                    for r, v in zip(routes, vectors)]
            execute_values(
                cur,
                "INSERT INTO routes (project_id, path, method, name, embedding) VALUES %s",
                data,
            )
            conn.commit()
        finally:
            cur.close()
            conn.close()

    def save_scenarios(self, project_id: str, scenarios: List, vectors: List) -> None:
        conn = self.get_conn()
        cur  = conn.cursor()
        try:
            data = [(project_id, s.description, s.input_json, v)
                    for s, v in zip(scenarios, vectors)]
            execute_values(
                cur,
                "INSERT INTO scenarios (project_id, description, input_json, embedding) VALUES %s",
                data,
            )
            conn.commit()
        finally:
            cur.close()
            conn.close()

    def save_code_context(
        self,
        project_id: str,
        chunks: List[Dict],
        vectors: List[List[float]],
    ) -> None:
        conn = self.get_conn()
        cur  = conn.cursor()
        try:
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
            conn.commit()
        finally:
            cur.close()
            conn.close()

    # ------------------------------------------------------------------
    # Recherche
    # ------------------------------------------------------------------

    def find_closest_route(self, project_id: str, query_vector: List[float]) -> Dict:
        """Trouve la route la plus similaire sémantiquement à la requête."""
        conn = self.get_conn()
        cur  = conn.cursor()
        try:
            cur.execute(
                """
                SELECT path, name, 1 - (embedding <=> %s::vector) AS similarity
                FROM   routes
                WHERE  project_id = %s
                ORDER  BY embedding <=> %s::vector
                LIMIT  1;
                """,
                (query_vector, project_id, query_vector),
            )
            row = cur.fetchone()
            if row:
                return {"path": row[0], "name": row[1], "score": float(row[2])}
            return None
        finally:
            cur.close()
            conn.close()

    def find_closest_code_context(
        self,
        project_id: str,
        query_vector: List[float],
        limit: int = 8,
    ) -> List[Dict]:
        """Retourne les N chunks de code les plus proches sémantiquement."""
        conn = self.get_conn()
        cur  = conn.cursor()
        try:
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
        finally:
            cur.close()
            conn.close()

    def get_code_by_class_name(self, project_id: str, class_name: str) -> List[Dict]:
        """Récupère les chunks correspondant à une classe par son nom (ILIKE)."""
        conn = self.get_conn()
        cur  = conn.cursor()
        try:
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
        finally:
            cur.close()
            conn.close()

    def get_project_stats(self, project_id: str) -> Dict:
        """Retourne des statistiques sur les données indexées d'un projet."""
        conn = self.get_conn()
        cur  = conn.cursor()
        try:
            cur.execute(
                "SELECT COUNT(*) FROM project_code_context WHERE project_id = %s",
                (project_id,),
            )
            chunk_count = cur.fetchone()[0]
            cur.execute(
                "SELECT COUNT(*) FROM routes WHERE project_id = %s",
                (project_id,),
            )
            route_count = cur.fetchone()[0]
            return {
                "project_id":  project_id,
                "chunks":      chunk_count,
                "routes":      route_count,
            }
        finally:
            cur.close()
            conn.close()
