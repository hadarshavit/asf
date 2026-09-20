# Metrics

Runtime scoring evaluates sequential schedules in order. Feature computation is
charged only when its step is reached; a successful presolver skips later steps.
Both feature and algorithm time count toward the scenario cutoff. A completion
exactly at the cutoff is accepted; otherwise an unsolved instance receives
`budget * par`.

For compatibility, schedules with no explicit feature steps pay the supplied
feature time upfront. A trailing block of multiple algorithms with identical
allocations of at least the scenario budget is treated as a parallel portfolio;
its elapsed algorithm time is the earliest completion. Feature costs still count
toward the cutoff. Interleaved feature and algorithm steps are evaluated sequentially.

::: asf.metrics
