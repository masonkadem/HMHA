# Direction 3: one redundancy law for synapses, vessels and heads (literature check, 2026-11-09)

Claim to test: keep one more redundant unit while
P(it works and too few others work) x (what it saves) >= its cost.
For synapses: a connection between two neurons uses N contacts; each fails to release with
probability 1 - Pr. The tree predicts more contacts where release is less reliable, given similar costs.

## What the literature already says (quick search, not a systematic review)
- Single contacts are unreliable: at CA1 contacts more than half of presynaptic spikes fail
  (Allen & Stevens 1994).
- Multiple contacts per connection are reported to improve reliability (thalamic afferents onto cortical
  interneurons, Bagnall et al. 2011; amphibian papilla vs mammalian cochlea, J Neurosci 2014).
- A modelling study finds redundancy compensates unreliability only if spike timing carries information.
- Release sites do not always equal anatomical contacts (barrel cortex L4->L2/3: equal; layer 5: release
  sites can exceed contacts by an order of magnitude), so "N" must be defined carefully.
- So the qualitative idea (redundancy offsets unreliable release) is known. What would be new is the
  quantitative cost-benefit rule predicting N from Pr and a cost, shared with vessels and attention heads.

## Concrete test (next step, about a day)
Use per-pathway tables of mean contacts per connection and release probability (U) from the Blue Brain
neocortical microcircuit (Markram et al. 2015; synaptic properties portal). Prediction: across pathways,
lower U goes with more contacts per connection. Then fit the tree with one cost parameter and check
whether a single cost explains the spread. Caveat: those values are partly model-derived.

## Sources
- https://bbp.epfl.ch/nmc-portal/synaptic-properties.html
- https://pmc.ncbi.nlm.nih.gov/articles/PMC2825095
- https://pmc.ncbi.nlm.nih.gov/articles/PMC45023
- https://sites.wustl.edu/bagnall/files/2021/05/2011_Bagnall.pdf
- https://arxiv.org/pdf/1609.05730
