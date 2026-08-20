<?php
/**
 * Pont AST pour code_parser.py.
 *
 * Lit le code PHP sur stdin, parse via nikic/php-parser, et émet sur stdout
 * un JSON décrivant la première classe/interface/trait trouvée : ses
 * attributs de classe, et pour chaque méthode ses attributs (#[Route],
 * #[IsGranted], ...) avec arguments résolus récursivement (gère un nombre
 * arbitraire de niveaux d'imbrication de tableaux — contrairement aux regex
 * à profondeur fixe qu'il remplace), ses paramètres typés, son type de
 * retour, son docblock brut, et le texte source exact de son corps.
 *
 * Erreur de parsing PHP → message sur stderr + exit 1 (même idiome que
 * validate_php_syntax() dans php_writer.py).
 */

require __DIR__ . '/vendor/autoload.php';

use PhpParser\ParserFactory;
use PhpParser\Node;
use PhpParser\NodeFinder;

$code = file_get_contents('php://stdin');

$parser = (new ParserFactory())->createForNewestSupportedVersion();
try {
    $ast = $parser->parse($code);
} catch (Throwable $e) {
    fwrite(STDERR, "Parse error: " . $e->getMessage() . "\n");
    exit(1);
}

if ($ast === null) {
    echo json_encode(null);
    exit(0);
}

/**
 * Résout récursivement une expression PHP en valeur JSON-compatible.
 * Gère les strings, entiers, booléens, tableaux imbriqués à N niveaux
 * (clé => valeur ou positionnels), et les constantes de classe
 * (Foo::class, Enum::CASE) sous forme de chaîne "Foo::class".
 * Retourne null pour toute expression non triviale (variable, appel de
 * fonction, etc.) plutôt que d'échouer — cohérent avec l'esprit "best
 * effort" du parsing existant.
 */
function exprToValue($expr) {
    if ($expr instanceof Node\Scalar\String_) return $expr->value;
    if ($expr instanceof Node\Scalar\Int_) return $expr->value;
    if ($expr instanceof Node\Scalar\Float_) return $expr->value;
    if ($expr instanceof Node\Expr\ConstFetch) {
        $name = strtolower($expr->name->toString());
        if ($name === 'true') return true;
        if ($name === 'false') return false;
        if ($name === 'null') return null;
        return $expr->name->toString();
    }
    if ($expr instanceof Node\Expr\Array_) {
        $out = [];
        $isList = true;
        foreach ($expr->items as $item) {
            if ($item === null) continue;
            if ($item->key !== null) {
                $isList = false;
                $out[exprToValue($item->key)] = exprToValue($item->value);
            } else {
                $out[] = exprToValue($item->value);
            }
        }
        return $out;
    }
    if ($expr instanceof Node\Expr\ClassConstFetch) {
        $className = $expr->class instanceof Node\Name ? $expr->class->toString() : '?';
        $constName = $expr->name instanceof Node\Identifier ? $expr->name->toString() : '?';
        return $className . '::' . $constName;
    }
    return null;
}

function attrGroupsToArray(array $attrGroups): array {
    $out = [];
    foreach ($attrGroups as $group) {
        foreach ($group->attrs as $attr) {
            $args = [];
            foreach ($attr->args as $arg) {
                $args[] = [
                    'name'  => $arg->name?->toString(),
                    'value' => exprToValue($arg->value),
                ];
            }
            $out[] = ['name' => $attr->name->toString(), 'args' => $args];
        }
    }
    return $out;
}

function docCommentText($node): ?string {
    $doc = $node->getDocComment();
    return $doc ? $doc->getText() : null;
}

/**
 * Convertit un noeud de type (Identifier, Name, NullableType, UnionType,
 * IntersectionType) en chaîne lisible ("?Foo", "Foo|Bar", "Foo&Bar").
 * NullableType/UnionType/IntersectionType n'implémentent PAS __toString(),
 * contrairement à Identifier/Name — cast direct impossible.
 */
function typeToString($type): ?string {
    if ($type === null) return null;
    if ($type instanceof Node\NullableType) {
        return '?' . typeToString($type->type);
    }
    if ($type instanceof Node\UnionType) {
        return implode('|', array_map('typeToString', $type->types));
    }
    if ($type instanceof Node\IntersectionType) {
        return implode('&', array_map('typeToString', $type->types));
    }
    // Identifier ou Name : implémentent __toString()
    return (string)$type;
}

function paramsToArray(array $params): array {
    $out = [];
    foreach ($params as $p) {
        $name = $p->var instanceof Node\Expr\Variable && is_string($p->var->name) ? $p->var->name : null;
        $out[] = [
            'name'        => $name,
            'type'        => typeToString($p->type),
            'has_default' => $p->default !== null,
        ];
    }
    return $out;
}

$finder = new NodeFinder();
$classLike = $finder->findFirstInstanceOf($ast, Node\Stmt\ClassLike::class);

if ($classLike === null) {
    echo json_encode(null);
    exit(0);
}

$methods = [];
foreach ($classLike->getMethods() as $method) {
    $bodyText = '';
    if ($method->stmts !== null && count($method->stmts) > 0) {
        $start = $method->stmts[0]->getStartFilePos();
        $end   = $method->stmts[count($method->stmts) - 1]->getEndFilePos();
        $bodyText = substr($code, $start, $end - $start + 1);
    }
    $methods[] = [
        'name'        => $method->name->toString(),
        'attributes'  => attrGroupsToArray($method->attrGroups),
        'params'      => paramsToArray($method->getParams()),
        'return_type' => typeToString($method->returnType),
        'docblock'    => docCommentText($method),
        'body'        => $bodyText,
    ];
}

$result = [
    'class_name'       => $classLike->name->toString(),
    'class_attributes' => attrGroupsToArray($classLike->attrGroups),
    'class_docblock'   => docCommentText($classLike),
    'methods'          => $methods,
];

echo json_encode($result, JSON_UNESCAPED_SLASHES);
