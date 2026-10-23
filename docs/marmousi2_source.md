# Marmousi2 public SEGY source

## Provenance

The public download page is:

```text
https://www.agl.uh.edu/downloads/downloads.htm
```

The files used here are:

```text
https://www.agl.uh.edu/downloads/segy_time/SYNTHETIC_time.segy.zip
https://www.agl.uh.edu/downloads/vp_marmousi-ii.segy.gz
```

The synthetic section is distributed by the Allied Geophysical Laboratories
at the University of Houston and is associated with the Marmousi2 elastic
model described by Martin, Wiley, and Marfurt (2006).

## File identity

| File | Dimensions | SHA-256 |
|---|---|---|
| `SYNTHETIC_time.segy.zip` | archive | `c5ae44815a6fcfaca75621e5a46abc781c75dddcbaffd9c18784489b00510ee6` |
| `SYNTHETIC_time.segy` | 2721 traces x 1001 samples, 4 ms | `3843344f58657f69377fad448bfd9c6f3d24f86ec3b367d04699759d3151de51` |
| `vp_marmousi-ii.segy.gz` | gzip archive | `2c7a0c4170c95a15620c2146573711c61a4cef8dc27338ef6c211a81ffa68920` |
| `vp_marmousi-ii.segy` | 13601 traces x 2801 samples | `2722a5de0645d8fcf5750f380893d4875383af9b24988ece0bd07b943f5b0c91` |

Default cache:

```text
$SEISPACE_DATA_ROOT/native_marmousi2
```

The files can be restored with `scripts/download_marmousi2_segy.sh`.

## Reporting boundary

`SYNTHETIC_time.segy` is a public synthetic seismic section generated from the
Marmousi2 model. It is not an original field recording and it is not the
original 1988 Marmousi shot gathers. It can be used as a public, reproducible
2D image benchmark, while the FORGE SEGY experiment remains the native field
observation benchmark.

For this section, use a trace split rather than a random patch split:

```text
train: first 60% of traces
validation: next 10% of traces
test: final 30% of traces
```

The test traces therefore never occur in training or validation patches.
Controlled STEAD ambient noise is added solely to create paired references.
Each patch is clipped at its `0.5` and `99.5` percentiles, standardized over
the complete block, and resampled if it is near-constant.
