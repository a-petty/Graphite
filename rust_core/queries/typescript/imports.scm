(import_statement source: (string) @import)
(export_statement source: (string) @import)
(call_expression
  function: (identifier) @_fn
  arguments: (arguments (string) @import)
  (#eq? @_fn "require")
)