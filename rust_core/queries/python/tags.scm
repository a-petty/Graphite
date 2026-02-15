; Function and Class definitions
(function_definition
  name: (identifier) @function.name
  parameters: (parameters) @function.params
  return_type: (type)? @function.return
  body: (block
    (expression_statement
      (string) @function.doc)?) @function.body
  (#set! "role" "definition"))

(class_definition
  name: (identifier) @class.name
  superclasses: (argument_list)? @class.bases
  body: (block)
  (#set! "role" "definition"))

; Import statements
(import_statement
  name: (dotted_name) @import.module)

(import_from_statement
  module_name: (dotted_name) @import.module
  name: (dotted_name) @import.symbol)
