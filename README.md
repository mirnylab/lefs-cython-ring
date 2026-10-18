# lefs-cython-ring

Circular-backbone support for [Mirnylab's lefs-cython](https://github.com/mirnylab/lefs-cython),
by Max Imakaev and Timothy Földes. The Cython simulator accepts genomic loading,
unloading, pausing and directional CTCF capture/release probabilities. Its
positions and status arrays can be used to measure cohesin occupancy, including
the subset of heads captured at CTCF.

Install from this checkout with `pip install -e .`. The Python import remains
`from lefs_cython.simple import LEFSimulator, constants`.

## Circular backbone

Pass `circular=True` to `LEFSimulator(...)` to connect sites `N-1` and `0`.
The default `circular=False` retains the original linear backbone.

- Every ring site is available for loading. The existing three-site empty
  loading window wraps across zero, and newly loaded loops retain their
  original one-site or two-site width distribution.
- Heads keep their directional identities as they cross zero. Leg 0 moves
  left and leg 1 moves right; their numerical indices need not be ordered.
  Occupied sites still block movement, including the same motor's other head.
- Capture, release, pausing and status-dependent unloading follow the same
  rules on both backbones. A released head can resume movement; changing the
  backbone does not change these probabilities or their time convention.
- The graph-distance helpers include the closing backbone arc. Coordinates
  and distances are in lattice-site units. A region with `start > end`
  crosses zero; `[0, N]` represents the whole ring. Call
  `populate_dijkstra_arrays(...)` after changing the motor configuration and
  before querying distances, as in the original API.

`circular=True` describes the one-dimensional genomic backbone. It does not
specify a three-dimensional periodic box or a polymer contact-distance rule.
The original architecture notes below describe the shared motor dynamics.

## Verification

Run `python -m pytest -q tests` to check both backbones. The 18 checks cover
the original linear dynamics plus ring crossings, loading and reloading at
zero, collisions, directional capture/release, and circular graph distances.

A CPU timing check with 50,000 sites and 435 motors took 1.63 seconds for
100,000 updates on the development host. It used 0.0216 s per decision,
125 bp/s per head at 1 kb/site, and 822-s residence. This measures forward
dynamics; obtaining a sufficiently sampled occupancy profile is a separate
requirement for fitting.

A uniform-ring control used nine motors on 1,000 sites and sampled 20,000
configurations over two million updates. Mean head occupancy over the ten
sites on either side of zero was 1.006 times that over the other sites.
This finite-sampling diagnostic found no special accumulation at the join.

## Original architecture notes

This example contains a draft of a new extrusion simulation code.

## Code architecture and assumption 

### Glossary 

#### Entities

* **LEF**:  that thing that extrudes loops. Cohesin in most cases. 
* **leg**: a "virtual leg" of cohesin that extrudes on one side. Whether or not cohesin has legs (or other body parts), if there is a loop from A to B, cohesin's "legs" are left at position A and right at position B 

#### processes 

* **loading**: a process of cohesin appearing on chromatin, by whatever means - we don't differentiate between targeted loading (e.g. at enhancers) and generic loading/binding/etc. anywhere along the sequence. 
* **unloading**: the opposite of loading. We don't differentiate active unloading and passive unloading. 
* **capture**: the process of cohesin becoming "captured" at a CTCF site and becoming a CTCF-bound (and possibly CTCF-stabilized) cohesin 
* **release**: the opposite of capture - cohesin's leg is released from CTCF and now is a normal moving leg. 

#### Conventions and internal variables

* **status**: one of ["moving", "paused", "captured"]   - a status of a cohesin leg. 
  * **moving** - means the leg was just created, just released, or previously successfully moved 
  * **paused** - means that the leg failed to move, because it got randomly paused 
  * **stalled** - means that the leg failed to move because something is in the way
    * A leg is always stalled if it is not captured and something is in the way 
  * **captured** - means that the leg is bound to CTCF and cannot move


#### Arrays 

* **LEFs** - a (NLEFs x 2) array of positions of two LEF legs 
* **statuses** - a (NLEFs x 4) array of statuses (moving, paused, stalled, captured) of each LEF leg

### Loading of LEFs to the sequence 

We assume random loading of a LEF with some load probability as a function of sequence.
Loading always happens upon unloading, so the probabilities are summed and re-normalized to sum to 1.
Note that actual loading will happen with a different distributions as some sites will be occupied more often than others.
To load, a LEF needs a width-3 open area. LEF either loads to positions [-1,0] or positions [-1,1] of the area 

As such, it is advisable to set loading probabilities of all CTCF sites to zero - so you can't load "across" a CTCF site. 
Also, extreme care needs to be taken with respect to what does it mean to "load" at enhancers. 
E.g. in cases with one-sided extrusion you may want one of the "ends" to remain at the enhancer while another to translocate away. 
In that case, you should modify the loading function carefully - e.g. you may forgo the width-2 loading at all. 

Width-2 loading is needed to avoid **checkering** in "naive" extrusion. 
Checkering would happen because if you load a LEF of width 1, only width 1, 3, 5, 7... would be possible until one of the two legs gets stopped at something, 
because we increase the LEF by 2 every step. 
Adding uniform pause probability of .25 will get rid of that checkering after 10-ish steps anyways, so maybe that is a solution. 
(in which case I would advice to load at (-1, 0) or (0,1) to be symmetric)
But for now, we are fighting checkering through width-2 loading, which would sometimes load "around" the loading site or a CTCF, but that's OK. 
We will never load around another LEF's leg as there is a check for that. 

### Unloading of LEFs from the sequence 

Each LEF leg will have a status (bound, moving, stalled) - and will have a corresponding Unload probability. 

The question is how to combine two legs' unloading probability into unloading for the whole LEF? 

The solution was made to take an average except when one of the two legs is stuck at CTCF ("bound"). 
When the leg is stuck at CTCF, it has a protective effect on the whole cohesin, including the other leg. 
So in that case we take the smallest unloadig probability. 


### Cohesin extrusion policies

Several cohesin leg movement policies are implemented, and can be assigned through the 'move_policy' argument.

**Symmetric extrusion ('move_policy=0')**
*Both legs are updated at each time step

**Alternate leg policy ('move_policy=1')**
* Only one leg moves at a time.
* If one leg is captured at CTCF site, the other leg is chosen to move.
* If neither leg is CTCF-bound, the updated leg alternates with a probability defined by `alternate_prob`.


**Importantly**
*If one leg is paused or stalled, the alternating logic remains, such that the other leg motion goes on and off at the same rate.
*Setting alternate_prob=0 enables 1 legged extrusion that switches sides when on leg is bound at CTCF.
