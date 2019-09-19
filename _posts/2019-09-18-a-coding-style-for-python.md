---
layout: post
title: A coding style for Python
categories: [Python, Coding, Zen]
---

    Beautiful is better than ugly.
        - PEP 20, The Zen of Python


## PEP 8

Python already have an official style guide, PEP 8. It's first doctrine:

"A Foolish Consistency is the Hobgoblin of Little Minds"

That is why I need an upgrade from PEP 8. 

For items not listed here, adhere to the PEP 8.

## Max Line Length

120 characters. 79 is too short.

## Closing Parentheses

Put closing parentheses at the same level of the last object. Do not put them on a new line.

```Python
x = dict(a=1, b=2,
    c=3, d=4, ) # closing parentheses same line, yes
```

```Python
x = dict(a=1, b=2,
    c=3, d=4
) # closing parentheses separate line, no
```

Rationale: Python structures code with indentations rather than parentheses.

## Indentation

Indent 1 level (4 spaces) for line continuation, or 2 levels to distinguish from the next line.
Never align with opening delimiter:

```Python
foo = very_long_function_name(arg,
    var_one=1, var_two=2, var_three=3, ) # 1 level indent, yes


foo = very_long_function_name(
    arg,
    var_one=1,
    var_two=2,
    var_three=3, ) # 1 level indent and align, yes


def very_long_function_name(arg, var_one=1, var_two=2,
        var_three=3, ): # 2 level indent to separate from the next line, yes
    implementation_starts_here = 0
```

```Python
# what if the function name plus indent is really long, like 70 characters?
foo = very_long_function_name(arg,
                              var_one=1,
                              var_two=2,
                              var_three=3) # random align, no
```

Rationale: Regularly structured code is beautiful. Caparicious alignment is ugly.

## Blank Lines

Completely avoid blank lines inside function and methods.
Instead, organize / refactor the code to be cleaner and shorter.
If there is a really strong need, use an empty comment at the same level of indentation.

```Python
if __name__ == '__main__':
    step_1_part_1() # no blank lines, yes
    step_1_part_2()
    step_2_part_1()
    step_2_part_2()
    step_3_part_3()


if __name__ == '__main__':
    step_1_part_1()
    step_1_part_2()
    #
    step_2_part_1() # blank comment to separate major blocks, ok
    step_2_part_2()
    step_3_part_3()
```

```Python
if __name__ == '__main__':
    step_1_part_1() # blank lines, no

    step_1_part_2()

    step_2_part_1()

    step_2_part_2()

    step_3_part_3()
```

Rationale: If you think the block is getting too dense, it probably means you need to refactor it.

## Inline comments

Keep them to a minimum.
Comments should reflect intentions, not echo implementaions.

Rationale: If the code is very readable, the comment is likely to be redundant.

## String Quotes

In code, use single-quote `''` whenever possible. For docstring, use triple double quotes: `""" """`.

```Python
if __name__ == '__main__': # single quote, yes
    # more stuff
```

```Python
if __name__ == "__main__": # double quote, no
    # more stuff
```
Rationale: Easier to type. Simple is better than complex.

## Type Annotations

Prefer detailed type information in the docstring than type annotations in the definitions.
However, if the toolset can provide useful functionalities from type annotation, it's absolutly ok to use.

Rationale: docstring is meant to be read by end users.

## Naming Conventions

If you don't want others to mess around with an object, add a single underscore before its name:

```Python
_do_not_mess_with_me
```

If a function / method alters its inputs, add a single underscore after it:

```Python
i_mess_with_inputs_(x) # content of x will change after each call
```
