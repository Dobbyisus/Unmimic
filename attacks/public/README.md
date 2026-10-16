# Public attack families

Published extraction-attack families, one YAML file per family. The input guard learns
from them, and the self red-team starts from them.

```yaml
family: verbatim_recall          # unique name; held-out evaluation splits by family
description: One sentence on what the attack tries.
examples:
  - "An example prompt."
```

Only older or summarized families go here. The full, current corpus lives in
`attacks/private/` (same format), which is gitignored and must never be committed: it is
dual-use. The input guard loads both folders.
