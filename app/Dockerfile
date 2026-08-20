FROM python:3.11-slim

# php-cli est requis par validate_php_syntax() pour `php -l` et par le pont
# AST (php_ast/parse.php, code_parser.py) qui remplace le parsing PHP par
# regex pour les routes/attributs #[Route]/#[IsGranted]. nikic/php-parser
# (composer.json de php_ast/) ne requiert que ext-tokenizer/json/ctype —
# des extensions du coeur PHP déjà incluses avec php-cli, pas de paquet
# apt séparé nécessaire.
RUN apt-get update \
 && apt-get install -y --no-install-recommends php-cli unzip \
 && rm -rf /var/lib/apt/lists/*

# Composer, pour installer nikic/php-parser (pont AST) — idiome standard
# multi-stage, pas d'installateur shell à télécharger.
COPY --from=composer:2 /usr/bin/composer /usr/bin/composer

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=600 -r requirements.txt

# Dépendances PHP du pont AST — copiées avant COPY . . pour profiter du
# cache de layers Docker (même logique que requirements.txt ci-dessus).
COPY php_ast/composer.json php_ast/composer.lock ./php_ast/
RUN cd php_ast && composer install --no-dev --optimize-autoloader --no-interaction

COPY . .

# On lance l'API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]