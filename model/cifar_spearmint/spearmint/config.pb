language: PYTHON
name:     "spearmint_solver"

variable {
 name: "CONV_OUTPUT"
 type: INT
 size: 3
 min:  10
 max:  80
}

variable {
 name: "CONV_SIZE"
 type: INT
 size: 3
 min:  2
 max:  5
}

# Integer example
#
# variable {
#  name: "Y"
#  type: INT
#  size: 5
#  min:  -5
#  max:  5
# }

# Enumeration example
# 
# variable {
#  name: "Z"
#  type: ENUM
#  size: 3
#  options: "foo"
#  options: "bar"
#  options: "baz"
# }


