Data organization
-----------------

*It is probably better to not touch the JSON files without using the wrappers generated in datahelpers. 
Otherwise, all data will be loaded into memory!*

```
N-Size
   |
  \/
Repetition
    |
   \/
Event(entry/exit)
```

Events are keyed with a string of form `str(frozenset[int])`.
therefore, to retrieve a frozenset object, evaluate each element of the index
per Event.

In general, events are of form:

```
frozenset([int]): int
frozenset([int]): int
.
.
.

```
