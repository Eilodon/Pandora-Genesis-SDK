//! Grid World simulation for testing Active Inference agents.
//!
//! A 2D grid environment with:
//! - Partial observability (agent can only see within a radius)
//! - Walls, goals, keys, doors
//! - Simple action model (move, pickup, use, wait)
//!
//! # Use Case
//! Unit test active inference agents without real sensors.
//! Verify belief updates, policy selection, and goal-directed behavior.

use thiserror::Error;

/// Errors that can occur in grid world simulation.
#[derive(Debug, Error)]
pub enum SimulationError {
    #[error("Invalid grid dimensions")]
    InvalidDimensions,
    #[error("Position out of bounds: ({0}, {1})")]
    OutOfBounds(i32, i32),
}

/// Types of cells in the grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    Empty,
    Wall,
    Key,
    Door(bool), // true = unlocked
    Goal,
}

/// Actions the agent can take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Move(Direction),
    Pickup,
    Use,
    Wait,
}

/// Movement directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    North,
    South,
    East,
    West,
}

impl Direction {
    /// Convert direction to (dx, dy) delta.
    pub fn to_delta(self) -> (i32, i32) {
        match self {
            Direction::North => (0, -1),
            Direction::South => (0, 1),
            Direction::East => (1, 0),
            Direction::West => (-1, 0),
        }
    }

    /// Get all four directions.
    pub fn all() -> [Direction; 4] {
        [Direction::North, Direction::South, Direction::East, Direction::West]
    }
}

/// Observability mode for the grid.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ObservabilityMode {
    /// Agent can see entire grid
    Full,
    /// Agent can only see within Manhattan radius
    Partial { range: i32 },
}

/// Viewshed tracking for partial observability.
#[derive(Default, Debug)]
pub struct Viewshed {
    /// Currently visible tile positions
    pub visible_tiles: Vec<(i32, i32)>,
    /// View range
    pub range: i32,
    /// Whether viewshed needs recalculation
    pub dirty: bool,
}

/// 2D grid world environment.
pub struct GridWorld {
    grid: Vec<Cell>,
    width: usize,
    height: usize,
    /// Current agent position
    pub agent_pos: (i32, i32),
    /// Goal position
    pub goal_pos: (i32, i32),
    /// Observability mode
    pub observability: ObservabilityMode,
    /// Viewshed state
    pub viewshed: Viewshed,
    /// Whether agent has a key
    has_key: bool,
}

impl GridWorld {
    /// Create a new grid world.
    ///
    /// Agent starts at (1, 1), goal is at (width-2, height-2).
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
            has_key: false,
        }
    }

    /// Create a simple test maze.
    pub fn simple_maze(width: usize, height: usize, mode: ObservabilityMode) -> Self {
        let mut world = Self::new(width, height, mode);
        
        // Add border walls
        for x in 0..width {
            world.place_wall((x, 0));
            world.place_wall((x, height - 1));
        }
        for y in 0..height {
            world.place_wall((0, y));
            world.place_wall((width - 1, y));
        }
        
        world
    }

    #[inline(always)]
    fn xy_idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    /// Place a wall at the given position.
    pub fn place_wall(&mut self, pos: (usize, usize)) {
        if pos.0 < self.width && pos.1 < self.height {
            let idx = self.xy_idx(pos.0, pos.1);
            self.grid[idx] = Cell::Wall;
        }
    }

    /// Place a key at the given position.
    pub fn place_key(&mut self, pos: (usize, usize)) {
        if pos.0 < self.width && pos.1 < self.height {
            let idx = self.xy_idx(pos.0, pos.1);
            self.grid[idx] = Cell::Key;
        }
    }

    /// Place a door at the given position.
    pub fn place_door(&mut self, pos: (usize, usize), unlocked: bool) {
        if pos.0 < self.width && pos.1 < self.height {
            let idx = self.xy_idx(pos.0, pos.1);
            self.grid[idx] = Cell::Door(unlocked);
        }
    }

    /// Get cell at position.
    pub fn get_cell(&self, x: i32, y: i32) -> Option<Cell> {
        if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
            Some(self.grid[self.xy_idx(x as usize, y as usize)])
        } else {
            None
        }
    }

    /// Update visibility based on current agent position.
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
                        // Euclidean distance check
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

    /// Get world state as 2D grid of cell codes.
    ///
    /// Cell codes:
    /// - 0: Empty (or unknown in partial observability)
    /// - 1: Wall
    /// - 2: Key
    /// - 3: Locked door
    /// - 4: Unlocked door
    /// - 5: Goal
    /// - 10: Agent
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

    /// Submit an action and get the result.
    pub fn submit_action(&mut self, action: Action) -> ActionResult {
        match action {
            Action::Move(dir) => {
                let (dx, dy) = dir.to_delta();
                let new_x = self.agent_pos.0 + dx;
                let new_y = self.agent_pos.1 + dy;

                // Bounds check
                if new_x < 0 || new_y < 0 || new_x >= self.width as i32 || new_y >= self.height as i32 {
                    return ActionResult::Collision;
                }

                let idx = self.xy_idx(new_x as usize, new_y as usize);
                
                // Wall check
                if self.grid[idx] == Cell::Wall {
                    return ActionResult::Collision;
                }

                // Locked door check
                if self.grid[idx] == Cell::Door(false) {
                    return ActionResult::Collision;
                }

                // Goal check
                if (new_x, new_y) == self.goal_pos {
                    self.agent_pos = (new_x, new_y);
                    self.viewshed.dirty = true;
                    return ActionResult::ReachedGoal;
                }

                self.agent_pos = (new_x, new_y);
                self.viewshed.dirty = true;
                ActionResult::Success
            }
            Action::Pickup => {
                let (x, y) = self.agent_pos;
                let idx = self.xy_idx(x as usize, y as usize);
                if self.grid[idx] == Cell::Key {
                    self.grid[idx] = Cell::Empty;
                    self.has_key = true;
                    ActionResult::PickedUpKey
                } else {
                    ActionResult::NothingToPickup
                }
            }
            Action::Use => {
                // Try to unlock adjacent doors
                for dir in Direction::all() {
                    let (dx, dy) = dir.to_delta();
                    let check_x = self.agent_pos.0 + dx;
                    let check_y = self.agent_pos.1 + dy;
                    
                    if check_x >= 0 && check_y >= 0 
                        && (check_x as usize) < self.width 
                        && (check_y as usize) < self.height 
                    {
                        let idx = self.xy_idx(check_x as usize, check_y as usize);
                        if self.grid[idx] == Cell::Door(false) && self.has_key {
                            self.grid[idx] = Cell::Door(true);
                            self.has_key = false;
                            return ActionResult::UnlockedDoor;
                        }
                    }
                }
                ActionResult::NothingToUse
            }
            Action::Wait => ActionResult::Success,
        }
    }

    /// Set agent position directly (for testing).
    pub fn set_agent_pos(&mut self, x: i32, y: i32) {
        self.agent_pos = (x, y);
        self.viewshed.dirty = true;
    }

    /// Check if agent has a key.
    pub fn has_key(&self) -> bool {
        self.has_key
    }

    /// Get grid dimensions.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Display grid as ASCII art.
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

    /// Calculate Manhattan distance from agent to goal.
    pub fn distance_to_goal(&self) -> i32 {
        (self.agent_pos.0 - self.goal_pos.0).abs() + (self.agent_pos.1 - self.goal_pos.1).abs()
    }
}

/// Result of an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionResult {
    Success,
    Collision,
    ReachedGoal,
    PickedUpKey,
    NothingToPickup,
    UnlockedDoor,
    NothingToUse,
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

        let _ = world.submit_action(Action::Move(Direction::East));
        assert!(world.viewshed.dirty);
        
        let _ = world.get_world_state();
        assert!(!world.viewshed.dirty);
    }

    #[test]
    fn agent_cannot_walk_through_walls() {
        let mut world = GridWorld::simple_maze(5, 5, ObservabilityMode::Full);
        world.set_agent_pos(1, 1);
        
        // Try to move north into wall
        let result = world.submit_action(Action::Move(Direction::North));
        assert_eq!(result, ActionResult::Collision);
        assert_eq!(world.agent_pos, (1, 1)); // Didn't move
    }

    #[test]
    fn agent_reaches_goal() {
        let mut world = GridWorld::new(5, 5, ObservabilityMode::Full);
        world.set_agent_pos(2, 2);
        world.goal_pos = (3, 2);
        
        let result = world.submit_action(Action::Move(Direction::East));
        assert_eq!(result, ActionResult::ReachedGoal);
        assert_eq!(world.agent_pos, (3, 2));
    }

    #[test]
    fn partial_observability_limits_vision() {
        let mut world = GridWorld::new(10, 10, ObservabilityMode::Partial { range: 2 });
        world.set_agent_pos(5, 5);
        
        let state = world.get_world_state();
        
        // Far corner should be unknown (0)
        assert_eq!(state[0][0], 0);
        
        // Agent position should be visible
        assert_eq!(state[5][5], 10);
    }

    #[test]
    fn key_and_door_mechanics() {
        let mut world = GridWorld::new(5, 5, ObservabilityMode::Full);
        world.set_agent_pos(2, 2);
        world.place_key((2, 2));
        world.place_door((3, 2), false);

        // Pick up key
        let result = world.submit_action(Action::Pickup);
        assert_eq!(result, ActionResult::PickedUpKey);
        assert!(world.has_key());

        // Unlock door
        let result = world.submit_action(Action::Use);
        assert_eq!(result, ActionResult::UnlockedDoor);
        assert!(!world.has_key());

        // Can now walk through door
        let result = world.submit_action(Action::Move(Direction::East));
        assert_eq!(result, ActionResult::Success);
    }

    #[test]
    fn display_shows_correct_characters() {
        let mut world = GridWorld::simple_maze(5, 5, ObservabilityMode::Full);
        world.set_agent_pos(2, 2);
        world.goal_pos = (3, 3);
        
        let display = world.display();
        assert!(display.contains('@')); // Agent
        assert!(display.contains('G')); // Goal
        assert!(display.contains('#')); // Walls
    }

    #[test]
    fn distance_to_goal_calculates_correctly() {
        let mut world = GridWorld::new(10, 10, ObservabilityMode::Full);
        world.set_agent_pos(2, 2);
        world.goal_pos = (5, 6);
        
        // Manhattan: |5-2| + |6-2| = 3 + 4 = 7
        assert_eq!(world.distance_to_goal(), 7);
    }
}
