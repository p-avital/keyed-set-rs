# Keyed Set: a hashbrown-based HashSet that indexes based on projections of its elements.
Ever wanted a `HashMap<K, V>`, but where `V` actually contains `K` (or at least can be projected to it)?
Well this is it.

The easiest way to define a projection is through a closure that you pass at construction, but you may also define your own key extractors as ZSTs that implement `Default` to gain a `Default` constructor for your Keyed Sets.