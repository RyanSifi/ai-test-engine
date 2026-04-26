import json
import os
import re

from code_parser import _find_method_block, _find_class_block


def find_method_code(class_name: str, method_name: str):
    """
    Parcourt src/ pour trouver la classe et extraire le corps de la méthode.
    Utilise le parser à accolades équilibrées du projet (pas de regex récursive).
    """
    project_root = "./src"
    for root, _, files in os.walk(project_root):
        if f"{class_name}.php" in files:
            file_path = os.path.join(root, f"{class_name}.php")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                print(f"Erreur lecture {file_path} : {e}")
                return None

            class_block = _find_class_block(content, class_name)
            if not class_block:
                print(f"Classe '{class_name}' non trouvée dans {file_path}.")
                return None

            method_block = _find_method_block(class_block, method_name)
            if not method_block:
                print(f"Méthode '{method_name}' non trouvée dans '{class_name}'.")
                return None

            return method_block.splitlines()

    print(f"Fichier '{class_name}.php' introuvable sous {project_root}.")
    return None


def main():
    print("Extracteur Automatique pour le Golden Dataset")
    class_name  = input("Nom de la classe (ex: ParcoursWorkflow) : ").strip()
    method_name = input("Nom de la méthode à extraire : ").strip()

    contexte = find_method_code(class_name, method_name)
    if not contexte:
        return

    print(f"Code extrait avec succès ({len(contexte)} lignes).")
    demande = input("\nDescription du test : ").strip()

    print("\nCollez le 'Test Idéal' (tapez 'FIN' sur une ligne seule pour valider) :")
    test_lines = []
    while True:
        line = input()
        if line.strip().upper() == "FIN":
            break
        test_lines.append(line)

    new_entry = {
        "demande_utilisateur": demande,
        "contexte_pertinent":  contexte,
        "test_ideal":          "\n".join(test_lines),
    }

    save_to_json(new_entry)


def save_to_json(entry: dict):
    filename = "golden_dataset.json"
    data = []
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    data.append(entry)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nDonnées ajoutées au fichier {filename} !")


if __name__ == "__main__":
    main()
