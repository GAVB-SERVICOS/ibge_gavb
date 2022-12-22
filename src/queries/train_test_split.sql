SELECT
    *,
    CASE
    WHEN ABS(MOD(FARM_FINGERPRINT(TO_JSON_STRING(i)), 10)) < 8 THEN 'train'
    ELSE 'test'
    END data_split
FROM
    `{{source_table}}` as i