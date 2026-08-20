from typing import List, Optional
from pydantic import BaseModel, Field
from config import settings

# Contraintes communes pour bloquer les inputs farfelus avant qu'ils n'atteignent
# les colonnes VARCHAR(100) de la DB ou l'écriture de fichiers.
_PROJECT_ID_PATTERN = r"^[a-zA-Z0-9_.-]+$"


class LearnFromCodeRequest(BaseModel):
    project_id: str = Field(..., min_length=1, max_length=100, pattern=_PROJECT_ID_PATTERN)
    model_name: Optional[str] = Field(
        default=None,
        max_length=200,
        json_schema_extra={"examples": [settings.default_embedding_model]},
    )
    model_config = {"protected_namespaces": ()}

    def model_post_init(self, __ctx) -> None:
        if not self.model_name or self.model_name == "string":
            self.model_name = settings.default_embedding_model


class GenerateTestRequest(BaseModel):
    project_id: str = Field(..., min_length=1, max_length=100, pattern=_PROJECT_ID_PATTERN)
    description: str = Field(..., min_length=1, max_length=4000)
    test_name: Optional[str] = Field(default=None, max_length=200)
    class_name: Optional[str] = Field(default=None, max_length=255)
    model_name: Optional[str] = Field(default=None, max_length=200)
    deterministic: bool = False  # True = bypass LLM, génération depuis les chunks
    async_mode: bool = False    # True = retourne job_id immédiatement, poll /job/{id}/status
    # True (défaut) = génération LLM PAR ROUTE (un appel par route, anti-troncature).
    # False = ancien chemin LLM monolithique (fichier complet en un appel) — conservé
    # pour comparaison / repli. Sans effet si deterministic=True.
    per_route: bool = True
    # Le moteur écrit dans tests/, où se trouvent aussi les tests écrits à la main.
    # Par défaut il REFUSE (409) d'écraser un fichier qu'il n'a pas produit lui-même.
    # overwrite=True force le remplacement, après création d'une sauvegarde .bak.
    overwrite: bool = False
    # Court-circuite le contrôle du code généré (exécution de commandes, réseau
    # sortant, suppression de fichiers). À n'utiliser que sur faux positif avéré,
    # après avoir lu le code — cf. app/prompt_safety.py.
    allow_unsafe: bool = False
    model_config = {"protected_namespaces": ()}

    def model_post_init(self, __ctx) -> None:
        if not self.model_name:
            self.model_name = settings.default_embedding_model


class GenerateUnitTestRequest(BaseModel):
    project_id: str = Field(..., min_length=1, max_length=100, pattern=_PROJECT_ID_PATTERN)
    file_path: str = Field(..., min_length=1, max_length=500)
    class_name: str = Field(..., min_length=1, max_length=255)
    method_name: Optional[str] = Field(default=None, max_length=255)
    description: str = Field(..., min_length=1, max_length=4000)
    # Pas de `test_name` ici, contrairement à GenerateTestRequest : le nom du
    # fichier de test unitaire est toujours dérivé de class_name
    # (`{ClasseCourte}Test.php`). Le champ existait mais n'était jamais lu.
    model_name: Optional[str] = Field(default=None, max_length=200)
    async_mode: bool = False
    overwrite: bool = False      # cf. GenerateTestRequest.overwrite
    allow_unsafe: bool = False   # cf. GenerateTestRequest.allow_unsafe
    model_config = {"protected_namespaces": ()}

    def model_post_init(self, __ctx) -> None:
        if not self.model_name:
            self.model_name = settings.default_embedding_model


class ResetSchemaRequest(BaseModel):
    confirm: bool = False


class GenerateTestsBatchRequest(BaseModel):
    project_id: str = Field(..., min_length=1, max_length=100, pattern=_PROJECT_ID_PATTERN)
    class_names: List[str] = Field(..., min_length=1, max_length=50)
    deterministic: bool = True   # Défaut déterministe pour le lot (plus rapide, zéro hallucination)
    description_prefix: str = Field(
        default="Tests fonctionnels pour",
        max_length=200,
    )
