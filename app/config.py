from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    database_url: str
    ollama_url: str = "http://host.docker.internal:11434/api/generate"
    default_embedding_model: str = 'paraphrase-multilingual-MiniLM-L12-v2'
    default_llm_model: str = 'qwen2.5-coder:7b'
    # Allowlist de modèles d'embedding chargeables via /learn-from-code et /generate-test.
    # Empêche un appelant de forcer le téléchargement d'un modèle arbitraire depuis HuggingFace.
    # Format CSV. Le default_embedding_model est toujours autorisé implicitement.
    allowed_embedding_models: str = (
        "paraphrase-multilingual-MiniLM-L12-v2,"
        "all-MiniLM-L6-v2,"
        "all-mpnet-base-v2"
    )
    container_project_root: str = "/workspace"

    # Clé API requise sur les endpoints d'écriture (header X-API-Key).
    # Si vide, l'authentification est désactivée (utile en dev). À DÉFINIR en prod.
    api_key: Optional[str] = None

    # CORS — origins autorisées, séparées par des virgules.
    # "*" autorise tout (à éviter en prod). Vide = CORS désactivé.
    cors_origins: str = ""

    # Contexte d'authentification injecté dans le prompt et le générateur déterministe.
    # Adapter ces valeurs à votre projet dans le .env si nécessaire.
    auth_firewall_name: str = "secured_area"         # nom du firewall Symfony (security.yaml)
    auth_redirect_path: str = "/connect/web-sso"     # URL de login (WebSSO, /login, etc.)
    auth_redirect_status: int = 307                  # code HTTP de la redirection non-auth
    auth_test_roles: str = "ADMIN,CONSULT"           # rôles disponibles dans TestUserFactory
    # Mapping rôle Symfony → clé TestUserFactory (ex: "ROLE_PARCOURS:PARCOURS,ROLE_ADMIN:ADMIN")
    # Laisser vide si les rôles du chunk correspondent déjà aux clés de la factory.
    auth_role_key_map: str = ""
    auth_test_class: str = "App\\Tests\\Security\\TestUserFactory"
    auth_sso_user_class: str = "Cnam\\WebSSOBundle\\User\\WebSSOUser"

settings = Settings()
