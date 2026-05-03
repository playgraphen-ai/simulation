# Real City

City-builder prototype in Bevy 0.18, inspired by Cities: Skylines but
optimised with a GPU-first data model: every person, road segment, and
building is stored as a row in an `Rgba32Float` storage texture. The
simulation runs in compute shaders and the CPU only authors (UI tools,
zoning) and renders.

## Controls

- **Right-click / middle-click drag** — orbit the camera
- **WASD** — pan
- **Mouse wheel** — zoom
- **Left-click** — apply the active tool on the tile under the cursor
- Right-side panel:
  - *Build* tools: Road, Zone Residential, Zone Office, Zone Shop, None
  - *+10 people* — spawn 10 new citizens
  - *Activities* sliders — duration of each activity and P(home→work)

## Gameplay loop

1. Paint residential, office and shop zones.
2. Paint roads. Zoned tiles adjacent to a road auto-materialise as level-0
   buildings.
3. Click *+10 people*. People pick a random home (residential) and work
   (office), then travel between the two by car, stopping for shopping in
   between with the probability set by the slider.
4. Buildings that saturate level up; chronic vacancies abandon them back to
   empty zones.

## Architecture

```
src/
  main.rs              wires plugins
  sim/                 CPU shadow of the GPU data textures
    grid.rs            tile state (Empty, Zone, Road, Building)
    people.rs          PersonRow × PEOPLE_CAPACITY  (Rgba32Float)
    roads.rs           RoadRow  × ROAD_CAPACITY    (Rgba32Float)
    buildings.rs       BuildingRow × BUILDING_CAPACITY (Rgba32Float)
    textures.rs        creates + uploads the three data textures
    counters.rs        HUD tallies
  render/              rendering — GLTF scenes from Kenney City Kit
  ui/                  bevy_ui tool palette, HUD, sliders
  compute/             simulation tick + pathfinding + path cache
    cpu_sim.rs         CPU mirror of sim_people.wgsl (the fast path)
    pathcache.rs       hot-segment cache + suffix reuse
    spawn.rs           handles SpawnPeopleRequest
assets/shaders/        sim_people.wgsl and pathfind.wgsl (GPU target)
assets/models/         CC0 models from Kenney (buildings/roads/vehicles)
```

The CPU simulation in `compute/cpu_sim.rs` is a 1:1 mirror of the
`assets/shaders/sim_people.wgsl` compute shader — same inputs, same state
transitions, same path-progression — so porting to GPU is a drop-in swap
later without changing gameplay.

## Data-texture schema (Rgba32Float)

| Texture | Texels per row | Fields |
|---|---|---|
| people   | 2 | `money age destination home` · `work activity activity_time path_cursor` |
| roads    | 2 | `ax ay bx by` · `speed_mean conn_a_packed conn_b_packed length` |
| buildings | 2 | `x y btype level` · `income occupants capacity road_seg` |

`conn_*_packed` stores up to 3 neighbour segment ids as
`count<<24 | id0<<16 | id1<<8 | id2`.

## Pathfinding

Segment-level BFS. Executed on the CPU today (`cpu_sim::bfs_segments`),
budgeted to 64 paths/frame so crowds don't stall the frame. The GPU version
(`assets/shaders/pathfind.wgsl`) runs one workgroup per path request; repath
happens every `REPATH_INTERVAL` seconds OR every `REPATH_INTERSECTIONS`
segments a person has cleared.

Hot segments (top 32 by usage, decayed each second) are cached in
`PathCache.suffix_cache`; a future BFS starting on one can splice the tail
instead of re-exploring.

## Building

This project uses the GNU Rust toolchain; the linker is pinned in
`.cargo/config.toml` to the WinLibs MinGW gcc installed via winget at
`MartinStorsjo.LLVM-MinGW.UCRT` + `BrechtSanders.WinLibs.MCF.UCRT`.

```
cargo run --release
```

Asset licence: Kenney City Kit and Car Kit, CC0.
