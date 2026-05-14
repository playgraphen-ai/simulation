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
                "Frame: {:.1} ms\nCompute: {:.1} ms\nRender: {:.1} ms\nCPU: {:.1} %\nRAM: {:.0} MB\nEntities: {}",
                frame_time, timings.last_compute_ms, timings.last_render_ms, cpu, ram, entities
            );
        } else {
            node.display = Display::None;
        }
    }
}
