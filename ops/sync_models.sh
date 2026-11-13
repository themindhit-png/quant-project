#!/usr/bin/env bash
# Копирование моделей репо в каталог бота БЕЗ перезаписи живых моделей, которые отличаются от репо (после переобучения ops/retrain.py они новее репо).
#   bash ops/sync_models.sh SRC_DIR DST_DIR [FORCE=0|1]
# Нет файла в DST -> копируется. Совпадает -> ничего. Отличается -> НЕ трогается (печатается предупреждение), если FORCE=1 (ops/install.sh --models) —
# заменяется после бэкапа DST.bak.<время>. Та же проверка, что в ops/deploy.sh.
set -euo pipefail
SRC="${1:?SRC}"; DST="${2:?DST}"; FORCE="${3:-0}"; TS="$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$DST"
if [[ "$FORCE" == 1 ]]; then
  for f in "$SRC"/*; do b="$(basename "$f")"; if [[ -f "$DST/$b" ]] && ! cmp -s "$f" "$DST/$b"; then cp -a "$DST" "$DST.bak.$TS"; echo "бэкап моделей: $DST.bak.$TS"; break; fi; done
fi
for f in "$SRC"/*; do
  b="$(basename "$f")"
  if [[ ! -f "$DST/$b" ]]; then cp -f "$f" "$DST/$b"; echo "модель добавлена: $b"
  elif cmp -s "$f" "$DST/$b"; then :
  elif [[ "$FORCE" == 1 ]]; then cp -f "$f" "$DST/$b"; echo "модель заменена (--models): $b"
  else echo "!! живая модель отличается от репо, НЕ перезаписана: $DST/$b (заменить из репо — только с ключом --models)"
  fi
done
