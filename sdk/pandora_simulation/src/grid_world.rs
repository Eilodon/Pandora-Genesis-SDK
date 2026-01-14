use pandora_error::PandoraError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    Empty,
    Wall,
    Key,
    Door(bool),
    Goal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Move(Direction),
    Pickup,
    Use,
    Wait,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    North,
    South,
    East,
    West,
}

impl Direction {
    pub fn to_delta(self) -> (i32, i32) {
        match self {
            Direction::North => (0, -1),
            Direction::South => (0, 1),
            Direction::East => (1, 0),
            Direction::West => (-1, 0),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ObservabilityMode {
    Full,
    Partial { range: i32 },
}

#[derive(Default)]
pub struct Viewshed {
    pub visible_tiles: Vec<(i32, i32)>,
    pub range: i32,
    pub dirty: bool,
}

pub struct GridWorld {
    grid: Vec<Cell>, // flat storage
    width: usize,
    height: usize,
    pub agent_pos: (i32, i32),
    pub goal_pos: (i32, i32),
    pub observability: ObservabilityMode,
    pub viewshed: Viewshed,
}

impl GridWorld {
    pub fn new(width: usize, height: usize, mode: ObservabilityMode) -> Self {
        Self {
            grid: vec![Cell::Empty; width * height],
            width,
            height,
            agent_pos: (1, 1),
            goal_pos: (width as i32 - 2, height as i32 - 2),
            observability: mode,
            viewshed: Viewshed {
                visible_tiles: Vec::new(),
                range: match mode {
                    ObservabilityMode::Full => width.max(height) as i32,
                    ObservabilityMode::Partial { range } => range,
                },
                dirty: true,
            },
        }
    }

    #[inline(always)]
    fn xy_idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    pub fn place_wall(&mut self, pos: (usize, usize)) {
        if pos.0 < self.width && pos.1 < self.height {
            let idx = self.xy_idx(pos.0, pos.1);
            self.grid[idx] = Cell::Wall;
        }
    }

    pub fn update_visibility(&mut self) {
        if !self.viewshed.dirty {
            return;
        }
        self.viewshed.dirty = false;
        self.viewshed.visible_tiles.clear();

        match self.observability {
            ObservabilityMode::Full => {
                for y in 0..self.height as i32 {
                    for x in 0..self.width as i32 {
                        self.viewshed.visible_tiles.push((x, y));
                    }
                }
            }
            ObservabilityMode::Partial { range } => {
                let (ax, ay) = self.agent_pos;
                for dy in -range..=range {
                    for dx in -range..=range {
                        if dx * dx + dy * dy <= range * range {
                            let x = ax + dx;
                            let y = ay + dy;
                            if x >= 0 && x < self.width as i32 && y >= 0 && y < self.height as i32 {
                                self.viewshed.visible_tiles.push((x, y));
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn get_world_state(&mut self) -> Vec<Vec<u8>> {
        self.update_visibility();
        let mut state = vec![vec![0u8; self.width]; self.height];

        for y in 0..self.height as i32 {
            for x in 0..self.width as i32 {
                let idx = self.xy_idx(x as usize, y as usize);
                let visible = match self.observability {
                    ObservabilityMode::Full => true,
                    ObservabilityMode::Partial { .. } => {
                        self.viewshed.visible_tiles.contains(&(x, y))
                    }
                };
                if visible {
                    state[y as usize][x as usize] = match self.grid[idx] {
                        Cell::Empty => 0,
                        Cell::Wall => 1,
                        Cell::Key => 2,
                        Cell::Door(false) => 3,
                        Cell::Door(true) => 4,
                        Cell::Goal => 5,
                    };
                    if (x, y) == self.agent_pos {
                        state[y as usize][x as usize] = 10;
                    }
                }
            }
        }
        state
    }

    pub fn submit_action(&mut self, action: Action) -> Result<ActionResult, PandoraError> {
        match action {
            Action::Move(dir) => {
                let (dx, dy) = dir.to_delta();
                let new_x = self.agent_pos.0 + dx;
                let new_y = self.agent_pos.1 + dy;

                // Bounds
                if new_x < 0
                    || new_y < 0
                    || new_x >= self.width as i32
                    || new_y >= self.height as i32
                {
                    return Ok(ActionResult::Collision);
                }

                // Wall
                let idx = self.xy_idx(new_x as usize, new_y as usize);
                if self.grid[idx] == Cell::Wall {
                    return Ok(ActionResult::Collision);
                }

                // Goal
                if (new_x, new_y) == self.goal_pos {
                    self.agent_pos = (new_x, new_y);
                    self.viewshed.dirty = true;
                    return Ok(ActionResult::ReachedGoal);
                }

                self.agent_pos = (new_x, new_y);
                self.viewshed.dirty = true;
                Ok(ActionResult::Success)
            }
            Action::Wait => Ok(ActionResult::Success),
            Action::Pickup | Action::Use => Ok(ActionResult::Success), // đơn giản hóa
        }
    }

    pub fn display(&self) -> String {
        let mut output = String::new();
        for y in 0..self.height as i32 {
            for x in 0..self.width as i32 {
                let ch = if (x, y) == self.agent_pos {
                    '@'
                } else if (x, y) == self.goal_pos {
                    'G'
                } else {
                    match self.grid[self.xy_idx(x as usize, y as usize)] {
                        Cell::Empty => '.',
                        Cell::Wall => '#',
                        Cell::Key => 'K',
                        Cell::Door(false) => 'D',
                        Cell::Door(true) => 'd',
                        Cell::Goal => 'G',
                    }
                };
                output.push(ch);
            }
            output.push('\n');
        }
        output
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionResult {
    Success,
    Collision,
    ReachedGoal,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewshed_updates_on_move() {
        let mut world = GridWorld::new(10, 10, ObservabilityMode::Partial { range: 2 });
        assert!(world.viewshed.dirty);
        let _ = world.get_world_state();
        assert!(!world.viewshed.dirty);

        let _ = world.submit_action(Action::Move(Direction::East)).unwrap();
        assert!(world.viewshed.dirty);
        let _ = world.get_world_state();
        assert!(!world.viewshed.dirty);
    }
}
