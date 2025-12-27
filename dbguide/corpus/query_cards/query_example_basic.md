---
doc_type: QUERY_CARD
dialect: [mysql, redshift]
domain: [exemplo]
tables: []
tags: [getting_started]
---

# Exemplo básico de query card

## Intenção
Retornar uma contagem simples de registros, para servir como exemplo de estrutura de QUERY_CARD.

## Inputs esperados
- data range opcional (ex.: :dt_ini, :dt_fim)

## SQL (MySQL)
```sql
SELECT DATE(created_at) AS dia, COUNT(*) AS total_registros
FROM minha_tabela
WHERE (:dt_ini IS NULL OR created_at >= :dt_ini)
	AND (:dt_fim IS NULL OR created_at < :dt_fim)
GROUP BY 1
ORDER BY 1;
```

## Validação
- Conferir se a soma dos `total_registros` em um período bate com o total da tabela no mesmo intervalo.
