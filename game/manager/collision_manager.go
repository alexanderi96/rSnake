package manager

import (
	"snake-game/game/entity"
	"snake-game/game/types"
)

type CollisionManager struct {
	grid types.Grid
}

func NewCollisionManager(grid types.Grid) *CollisionManager {
	return &CollisionManager{
		grid: grid,
	}
}

// CheckCollision checks all types of collisions for a given position
func (cm *CollisionManager) CheckCollision(pos types.Point, snakes []*entity.Snake, currentSnake *entity.Snake) (bool, *entity.Snake) {
	// Check wall collision
	if cm.isWallCollision(pos) {
		return true, nil
	}

	// Check snake collisions
	if snake := cm.isSnakeCollision(pos, snakes, currentSnake); snake != nil {
		return true, snake
	}

	return false, nil
}

// isWallCollision checks if a position collides with walls
func (cm *CollisionManager) isWallCollision(pos types.Point) bool {
	return pos.X < 0 || pos.X >= cm.grid.Width || pos.Y < 0 || pos.Y >= cm.grid.Height
}

// isSnakeCollision checks if a position collides with any snake
// Returns the snake that was collided with, or nil if no collision
func (cm *CollisionManager) isSnakeCollision(pos types.Point, snakes []*entity.Snake, currentSnake *entity.Snake) *entity.Snake {
	for _, snake := range snakes {
		if snake == nil || snake == currentSnake || snake.Dead {
			continue
		}

		// Check collision with snake's head
		head := snake.GetHead()
		if pos == head {
			return snake // Return the snake for potential reproduction
		}

		// Check collision with snake's body (excluding head)
		for i := 0; i < len(snake.Body)-1; i++ {
			if pos == snake.Body[i] {
				return snake
			}
		}
	}

	// Check self collision for current snake
	if currentSnake != nil {
		// Skip the head and last body segment when checking self-collision
		// This prevents false positives during normal movement
		for i := 0; i < len(currentSnake.Body)-2; i++ {
			if pos == currentSnake.Body[i] {
				return currentSnake
			}
		}
	}

	return nil
}

// ValidateSpawnPosition checks if a position is valid for spawning a new snake
func (cm *CollisionManager) ValidateSpawnPosition(pos types.Point, snakes []*entity.Snake) bool {
	// Check wall collision
	if cm.isWallCollision(pos) {
		return false
	}

	// Check collision with any snake
	for _, snake := range snakes {
		if snake == nil || snake.Dead {
			continue
		}

		// Check collision with entire snake body
		for _, bodyPart := range snake.Body {
			if pos == bodyPart {
				return false
			}
		}
	}

	return true
}

// CheckHeadToHeadCollision specifically checks for head-to-head collisions
// Returns true and the other snake if there's a head-to-head collision
func (cm *CollisionManager) CheckHeadToHeadCollision(pos types.Point, snakes []*entity.Snake, currentSnake *entity.Snake) (bool, *entity.Snake) {
	for _, snake := range snakes {
		if snake == nil || snake == currentSnake || snake.Dead {
			continue
		}

		head := snake.GetHead()
		if pos == head {
			return true, snake
		}
	}

	return false, nil
}

// IsFoodCollision checks if a position collides with food
func (cm *CollisionManager) IsFoodCollision(pos types.Point, food types.Point) bool {
	return pos == food
}

// HandleMovement processes a snake's movement and checks all possible collisions
func (cm *CollisionManager) HandleMovement(snake *entity.Snake, newHead types.Point, snakes []*entity.Snake) (bool, *entity.Snake, bool) {
	// Returns: isDead, collidedSnake, isHeadToHead
	hasCollision, collidedSnake := cm.CheckCollision(newHead, snakes, snake)
	if hasCollision {
		if collidedSnake != nil {
			// Check if it's a head-to-head collision
			isHeadToHead, _ := cm.CheckHeadToHeadCollision(newHead, snakes, snake)
			if isHeadToHead {
				return true, collidedSnake, true
			}
		}
		return true, collidedSnake, false
	}
	return false, nil, false
}

// CheckFoodCollisions checks if a snake has collided with any food
func (cm *CollisionManager) CheckFoodCollisions(pos types.Point, foodList []types.Point) (bool, types.Point) {
	for _, food := range foodList {
		if cm.IsFoodCollision(pos, food) {
			return true, food
		}
	}
	return false, types.Point{}
}
