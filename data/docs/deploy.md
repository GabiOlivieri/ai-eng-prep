# Deploy

Para deployar a producción usá `acme deploy --env prod`. El deploy hace rollback automático
si el health check falla durante los primeros 3 minutos.

## Ambientes

Hay tres ambientes: dev, staging y prod. Solo los administradores pueden deployar a prod.
