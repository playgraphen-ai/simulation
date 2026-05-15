use bevy::{
    diagnostic::{
        DiagnosticsStore, EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin,
        SystemInformationDiagnosticsPlugin,
    },
    prelude::*,
};

#[derive(Component)]
pub struct PerfRoot;

#[derive(Component)]
pub struct FpsText;

#[derive(Component)]
pub struct DetailsText;

pub fn setup_perf_ui(mut commands: Commands) {
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(12.),
            right: Val::Px(12.),
            flex_direction: FlexDirection::Column,
            padding: UiRect::all(Val::Px(8.)),
            align_items: AlignItems::FlexEnd,
            ..default()
        },
        BackgroundColor(Color::srgba(0.1, 0.1, 0.15, 0.85)),
        Interaction::default(),
        PerfRoot,
    )).with_children(|p| {
        p.spawn((
            Text::new("FPS: ..."),
            TextFont { font_size: 14.0, ..default() },
            TextColor(Color::srgb(0.2, 1.0, 0.2)),
            FpsText,
        ));
        p.spawn((
            Text::new(""),
            TextFont { font_size: 12.0, ..default() },
            TextColor(Color::srgb(0.8, 0.8, 0.8)),
            Node {
                display: Display::None,
                margin: UiRect::top(Val::Px(4.)),
                ..default()
            },
            DetailsText,
        ));
    });
}

pub fn update_perf_ui(
    diagnostics: Res<DiagnosticsStore>,
    timings: Res<crate::DetailedTimings>,
    root_q: Query<&Interaction, With<PerfRoot>>,
    mut fps_q: Query<&mut Text, With<FpsText>>,
    mut details_q: Query<(&mut Text, &mut Node), (With<DetailsText>, Without<FpsText>)>,
) {
    let Ok(interaction) = root_q.single() else { return; };
    let hovered = matches!(*interaction, Interaction::Hovered | Interaction::Pressed);

    let fps = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS).and_then(|d| d.smoothed()).unwrap_or(0.0);
    
    if let Ok(mut text) = fps_q.single_mut() {
        text.0 = format!("FPS: {:.0}", fps);
    }

    if let Ok((mut text, mut node)) = details_q.single_mut() {
        if hovered {
            node.display = Display::Flex;
            let frame_time = diagnostics.get(&FrameTimeDiagnosticsPlugin::FRAME_TIME).and_then(|d| d.smoothed()).unwrap_or(0.0);
            
            let cpu = diagnostics.get(&SystemInformationDiagnosticsPlugin::SYSTEM_CPU_USAGE).and_then(|d| d.smoothed()).unwrap_or(0.0);
            let ram = diagnostics.get(&SystemInformationDiagnosticsPlugin::SYSTEM_MEM_USAGE).and_then(|d| d.smoothed()).unwrap_or(0.0);
            let entities = diagnostics.get(&EntityCountDiagnosticsPlugin::ENTITY_COUNT).and_then(|d| d.smoothed()).unwrap_or(0.0);
            
            text.0 = format!(
                "Frame: {:.1} ms\nLogic CPU: {:.1} ms\nCompute: {:.1} ms\n  - Logic: {:.3} ms (Peak: {:.3} ms)\n  - Bldgs: {:.3} ms (Peak: {:.3} ms)\n  - Roads: {:.3} ms (Peak: {:.3} ms)\n  - Path: {:.3} ms (Peak: {:.3} ms)\nRender: {:.1} ms\nCPU: {:.1} %\nRAM: {:.0} MB\nEntities: {}\n\nCycles:\n  - Logic: {}/{}\n  - Bldgs: {}/{}\n  - Roads: {}/{}\n  - Recount: {}/{}\n\nReadbacks:\n  - Stats: {:.1} ms\n  - Person: {:.1} ms\n  - Bldg: {:.1} ms\n\nThroughput:\n  - Logic: {:.0} /ms\n  - Path: {:.0} /ms\n\nBuffer Occupancy:\n  - People: {:.1}%\n  - Bldgs: {:.1}%\n  - Segments: {:.1}%",
                frame_time, 
                timings.smooth_update_ms, 
                timings.smooth_compute_ms, 
                timings.smooth_logic_ms, timings.peak_logic_ms,
                timings.smooth_bldg_ms, timings.peak_bldg_ms,
                timings.smooth_road_ms, timings.peak_road_ms,
                timings.smooth_pathfind_ms, timings.peak_pathfind_ms,
                timings.smooth_render_ms, 
                cpu, 
                ram, 
                entities,
                timings.logic_cycle.0, timings.logic_cycle.1,
                timings.bldg_cycle.0, timings.bldg_cycle.1,
                timings.road_cycle.0, timings.road_cycle.1,
                timings.recount_cycle.unwrap_or((0,0)).0, timings.recount_cycle.unwrap_or((0,0)).1,
                timings.rb_stats_ms,
                timings.rb_person_ms,
                timings.rb_bldg_ms,
                timings.throughput_logic,
                timings.throughput_path,
                timings.occ_people,
                timings.occ_buildings,
                timings.occ_segments,
            );
        } else {
            node.display = Display::None;
        }
    }
}
