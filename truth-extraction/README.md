1. understand the code in Burns22
2. Make it pretty generic: we're passing prompts to network and then looking at some
   hidden states.

Would be nice to split up dataset away so I can understand it. How are contrast prompts
generated? Also something I can run on my machine.

Dataloader stuff:
- What is `Dataset`? In pytorch right?
  - Map of keys to data samples. What's a sample? Wait what's a key?
- How does it do that, a bit lower level but not inside pytorch internals?
   - We have `ContrastDataset`, then get subset of it with `Subset`, then call `DataLoader`
     on `Subset`.
- What is `DataLoader`?
  - Seems like it samples from `Dataset` to give an iterable over items. What's an item?
    - We have this: `neg_ids, pos_ids, _, _, gt_label = batch`. So that's what an item
      in dataloader is.

Okay remaining questions:
- What is `ContrastDataset` doing super high level?
  - All methods are helpers for `__getitem__`, which returns this mysterious batch element.
  - What is `__getitem__` doing?
    - Gets original example from `self.raw_dataset`, which is just part (eg "test set")
      of `datasets.load_dataset()`
    - Get "text" and "label" from this original example.
      - Hard to see what these actually are... I know "label" options length is 2,
        and `neg_example` is `{"text": text, "label": 0}` and `pos_example` is same
        with `"label": 1`. But is `text` question, or question+answer?
    - `self.prompt` is single prompt from collection of prompts for given `raw_dataset`
      from `promptsource`. Let's find example of these prompts...
      - Okay, not 100% sure what this prompt looks like, but from notebook there's
        an example like:
        (text, label) => f"The following movie review expresses a {['positive', 'negative'][label]} sentiment:\n{text}"
    - Create a prompt for both negative and positive using `self.prompt`^
    - tokenize
    -
- Where are `neg_ids, pos_ids, _, _, gt_label` elements for each batch set? (Probably
  somewhere in `ContrastDataset`?)

Okay now let's migrate it to `godshatter`. Don't look too deeply into `Dataset` and
`pytorch` internals for this, document a little as I go.
- Interesting implementation note: seems like model type and information about decoder (?)
  is required for tokenizing properly. How can I make this a nicer split?
