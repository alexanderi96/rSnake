package main

import (
	"fmt"
	"math"

	"snake-game/qlearning"
)

// SnakeAgent rappresenta l'agente che gioca a Snake usando Q-learning.
type SnakeAgent struct {
	agent        *qlearning.Agent
	game         *Game
	foodReward   float64
	deathPenalty float64
}

// NewSnakeAgent crea un nuovo agente per il gioco.
func NewSnakeAgent(game *Game) *SnakeAgent {
	agent := qlearning.NewAgent(0.5, 0.8, 0.95) // Learning rate aumentato per apprendimento più rapido
	return &SnakeAgent{
		agent:        agent,
		game:         game,
		foodReward:   500.0,  // Default reward values
		deathPenalty: -250.0, // Can be overridden by curriculum
	}
}

// SetRewardValues imposta i valori di reward per la fase corrente
func (sa *SnakeAgent) SetRewardValues(foodReward, deathPenalty float64) {
	sa.foodReward = foodReward
	sa.deathPenalty = deathPenalty
}

// getState costruisce uno stato più dettagliato che include:
// - Direzione del cibo
// - Distanze dai muri e dal proprio corpo
// - Configurazione dei pericoli nelle 8 celle circostanti
// - Lunghezza del serpente normalizzata
// - Distanza normalizzata dal cibo
func (sa *SnakeAgent) getState() string {
	snake := sa.game.GetSnake()
	head := snake.GetHead()
	food := sa.game.food

	// Direzione del cibo (0=sopra, 1=destra, 2=sotto, 3=sinistra)
	foodDirX := food.X - head.X
	foodDirY := food.Y - head.Y
	foodDir := -1

	if math.Abs(float64(foodDirX)) > math.Abs(float64(foodDirY)) {
		if foodDirX > 0 {
			foodDir = 1 // destra
		} else {
			foodDir = 3 // sinistra
		}
	} else {
		if foodDirY > 0 {
			foodDir = 2 // giù
		} else {
			foodDir = 0 // su
		}
	}

	// Distanze dai pericoli nelle varie direzioni
	distAhead, distLeft, distRight := sa.game.GetDangers()

	// Distanza normalizzata dal cibo
	foodDist := math.Sqrt(math.Pow(float64(head.X-food.X), 2) + math.Pow(float64(head.Y-food.Y), 2))
	foodDistNorm := int(math.Min(foodDist/5.0, 10.0))

	// Lunghezza del serpente (normalizzata)
	length := math.Min(float64(len(snake.Body)), 20.0)
	lengthNorm := int(length / 4.0)

	// Configurazione pericoli vicini (8 celle circostanti)
	dangerN := sa.game.checkCollision(Point{X: head.X, Y: head.Y - 1}) != NoCollision
	dangerNE := sa.game.checkCollision(Point{X: head.X + 1, Y: head.Y - 1}) != NoCollision
	dangerE := sa.game.checkCollision(Point{X: head.X + 1, Y: head.Y}) != NoCollision
	dangerSE := sa.game.checkCollision(Point{X: head.X + 1, Y: head.Y + 1}) != NoCollision
	dangerS := sa.game.checkCollision(Point{X: head.X, Y: head.Y + 1}) != NoCollision
	dangerSW := sa.game.checkCollision(Point{X: head.X - 1, Y: head.Y + 1}) != NoCollision
	dangerW := sa.game.checkCollision(Point{X: head.X - 1, Y: head.Y}) != NoCollision
	dangerNW := sa.game.checkCollision(Point{X: head.X - 1, Y: head.Y - 1}) != NoCollision

	dangerPattern := fmt.Sprintf("%d%d%d%d%d%d%d%d",
		boolToInt(dangerN), boolToInt(dangerNE),
		boolToInt(dangerE), boolToInt(dangerSE),
		boolToInt(dangerS), boolToInt(dangerSW),
		boolToInt(dangerW), boolToInt(dangerNW))

	// Stato finale: combina tutte le informazioni
	state := fmt.Sprintf("%d:%d:%d:%d:%d:%d:%s:%d",
		int(sa.game.GetCurrentDirection()), foodDir,
		distAhead, distLeft, distRight,
		foodDistNorm, dangerPattern, lengthNorm)

	return state
}

// boolToInt converte un booleano in intero
func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

// relativeActionToAbsolute converte un'azione relativa in una direzione assoluta.
// Le azioni relative sono definite come:
//
//	0: ruota a sinistra
//	1: vai avanti
//	2: ruota a destra
func (sa *SnakeAgent) relativeActionToAbsolute(relativeAction int) Direction {
	currentDir := sa.game.GetCurrentDirection()
	switch relativeAction {
	case 0: // ruota a sinistra
		return currentDir.TurnLeft()
	case 1: // vai avanti
		return currentDir
	case 2: // ruota a destra
		return currentDir.TurnRight()
	default:
		return currentDir // fallback
	}
}

// Update esegue un passo di decisione e aggiornamento Q-learning.
func (sa *SnakeAgent) Update() {
	if sa.game.GetSnake().Dead {
		return
	}

	currentState := sa.getState()
	// Usa 3 possibili azioni relative.
	action := sa.agent.GetAction(currentState, 3)

	// Converte l'azione relativa in una direzione assoluta.
	newDir := sa.relativeActionToAbsolute(action).ToPoint()

	// Salva il punteggio corrente per calcolare il reward.
	oldScore := sa.game.GetSnake().Score
	oldLength := len(sa.game.GetSnake().Body)

	// Applica l'azione.
	sa.game.GetSnake().SetDirection(newDir)
	sa.game.Update()

	// Calcola il reward.
	reward := sa.calculateReward(oldScore, oldLength)

	// Aggiorna i Q-values.
	newState := sa.getState()
	sa.agent.Update(currentState, action, reward, newState, 3)
}

func (sa *SnakeAgent) calculateReward(oldScore, oldLength int) float64 {
	snake := sa.game.GetSnake()
	head := snake.GetHead()
	previousHead := snake.GetPreviousHead()
	food := sa.game.food

	// Reward base per essere ancora vivo (proporzionale alla lunghezza)
	reward := math.Min(float64(len(snake.Body))*0.2, 10.0)

	// --- REWARD PER IL CIBO ---

	// Reward per mangiare cibo (valore basato sulla fase di training)
	if snake.Score > oldScore {
		// Reward base per mangiare
		reward += sa.foodReward

		// Bonus proporzionale alla lunghezza attuale (incentiva crescita continua)
		reward += float64(len(snake.Body)) * 2.0

		// Reset del contatore di stagnazione
		sa.game.Steps = len(snake.Body) * 10
	}

	// --- REWARD PER AVVICINAMENTO AL CIBO ---

	// Calcola distanza precedente e attuale dal cibo
	oldDist := math.Sqrt(math.Pow(float64(previousHead.X-food.X), 2) +
		math.Pow(float64(previousHead.Y-food.Y), 2))
	newDist := math.Sqrt(math.Pow(float64(head.X-food.X), 2) +
		math.Pow(float64(head.Y-food.Y), 2))

	// Reward per avvicinamento con scala logaritmica (più significativo quando si è vicini)
	distDiff := oldDist - newDist
	if distDiff > 0 {
		// Avvicinamento
		reward += 10.0 * distDiff * (1.0 + 1.0/math.Max(newDist, 1.0))
	} else {
		// Allontanamento (penalità più leggera)
		reward += 5.0 * distDiff
	}

	// --- REWARD PER SPAZIO APERTO ---

	// Premiare quando il serpente mantiene più spazio aperto attorno a sé
	freedomReward := sa.calculateFreedomReward()
	reward += freedomReward

	// --- REWARD PER EVITARE TRAPPOLE ---

	// Penalizzare movimenti che portano a situazioni di "tunnel" o vicoli ciechi
	trapPenalty := sa.calculateTrapPenalty()
	reward -= trapPenalty

	// --- REWARD PER EFFICIENZA DEL PERCORSO ---

	// Premiare percorsi efficienti verso il cibo
	if len(snake.Body) > 5 {
		pathEfficiency := sa.calculatePathEfficiency()
		reward += pathEfficiency
	}

	// --- PENALITÀ PER LA MORTE ---

	if snake.Dead {
		// Penalità base per la morte
		reward = sa.deathPenalty

		// Penalità aggiuntiva proporzionale alla distanza dal cibo
		// (morte vicino al cibo è più grave)
		if newDist < 5 {
			reward -= (5 - newDist) * 10
		}

		// Penalità ridotta se il serpente è già molto lungo
		// (incentiva il rischio quando già cresciuto)
		if len(snake.Body) > 15 {
			reward *= 0.8
		}
	}

	// --- PENALITÀ PER STAGNAZIONE ---

	// Penalità per stagnazione con crescita più lenta e progressiva
	stepsWithoutFood := sa.game.Steps - oldLength*10
	if stepsWithoutFood > 30 {
		// Penalità esponenziale che cresce col tempo
		stagnationPenalty := math.Pow(float64(stepsWithoutFood-30)*0.05, 2)
		reward -= math.Min(stagnationPenalty, 150.0) // Cap massimo sulla penalità
	}

	// --- REWARD PER COMPORTAMENTI STRATEGICI ---

	// Premiare comportamenti perimetrali quando il serpente è lungo
	if len(snake.Body) > 10 {
		perimeterBonus := sa.calculatePerimeterBonus()
		reward += perimeterBonus
	}

	return reward
}

// calculateFreedomReward calcola un reward basato sullo spazio libero intorno alla testa
func (sa *SnakeAgent) calculateFreedomReward() float64 {
	head := sa.game.GetSnake().GetHead()
	freedom := 0

	// Controlla le 8 celle circostanti
	for dx := -1; dx <= 1; dx++ {
		for dy := -1; dy <= 1; dy++ {
			if dx == 0 && dy == 0 {
				continue // Salta la cella centrale (testa)
			}

			checkPoint := Point{X: head.X + dx, Y: head.Y + dy}
			if sa.game.checkCollision(checkPoint) == NoCollision {
				freedom++
			}
		}
	}

	// Reward esponenziale che premia di più avere più spazio libero
	return math.Pow(float64(freedom), 1.5)
}

// calculateTrapPenalty penalizza situazioni di "tunnel" o vicoli ciechi
func (sa *SnakeAgent) calculateTrapPenalty() float64 {
	head := sa.game.GetSnake().GetHead()
	grid := sa.game.Grid

	// Verifica se il serpente è in un corridoio con una sola via d'uscita
	adjacentWalls := 0
	possibleMoves := 0

	// Controlla le 4 direzioni cardinali
	directions := []Point{
		{X: 0, Y: -1}, // Nord
		{X: 1, Y: 0},  // Est
		{X: 0, Y: 1},  // Sud
		{X: -1, Y: 0}, // Ovest
	}

	for _, dir := range directions {
		checkPoint := Point{X: head.X + dir.X, Y: head.Y + dir.Y}

		// Controlla se è fuori dai limiti
		if checkPoint.X < 0 || checkPoint.X >= grid.Width ||
			checkPoint.Y < 0 || checkPoint.Y >= grid.Height {
			adjacentWalls++
			continue
		}

		// Controlla collisioni con il corpo
		if sa.game.checkCollision(checkPoint) != NoCollision {
			adjacentWalls++
		} else {
			possibleMoves++
		}
	}

	// Penalizza fortemente corridoi con una sola uscita
	if possibleMoves == 1 {
		// Controllo addizionale per vedere quanto è profondo il corridoio
		depth := sa.measureTrapDepth()
		return 20.0 + float64(depth)*5.0
	} else if possibleMoves == 2 && adjacentWalls == 2 {
		// Situazione a "L" - potenziale trappola
		return 10.0
	}

	return 0.0
}

// measureTrapDepth misura quanto è profondo un vicolo cieco
func (sa *SnakeAgent) measureTrapDepth() int {
	// Implementazione base - può essere estesa con BFS/DFS
	head := sa.game.GetSnake().GetHead()
	currentDir := sa.game.GetCurrentDirection().ToPoint()

	depth := 0
	currentPos := Point{X: head.X + currentDir.X, Y: head.Y + currentDir.Y}

	// Continua a muoversi in avanti finché possibile
	for sa.game.checkCollision(currentPos) == NoCollision {
		depth++
		currentPos.X += currentDir.X
		currentPos.Y += currentDir.Y

		// Prevenzione di loop infiniti
		if depth > 20 {
			break
		}
	}

	return depth
}

// calculatePathEfficiency premia percorsi efficienti verso il cibo
func (sa *SnakeAgent) calculatePathEfficiency() float64 {
	snake := sa.game.GetSnake()
	head := snake.GetHead()
	food := sa.game.food

	// Distanza diretta (Manhattan) dal cibo
	manhattanDist := float64(math.Abs(float64(head.X-food.X)) + math.Abs(float64(head.Y-food.Y)))

	// Percorso effettivo (numero di celle attraversate)
	actualPath := sa.estimateActualPath()

	// Rapporto di efficienza (1.0 = percorso ottimale)
	efficiency := math.Min(manhattanDist/math.Max(actualPath, 1.0), 1.0)

	return efficiency * 15.0
}

// estimateActualPath stima la lunghezza del percorso effettivo verso il cibo
func (sa *SnakeAgent) estimateActualPath() float64 {
	// Implementazione semplificata - in una versione completa
	// si potrebbe usare A* o altri algoritmi di pathfinding
	head := sa.game.GetSnake().GetHead()
	food := sa.game.food

	// Distanza Euclidea come approssimazione
	return math.Sqrt(math.Pow(float64(head.X-food.X), 2) + math.Pow(float64(head.Y-food.Y), 2))
}

// calculatePerimeterBonus premia comportamenti perimetrali quando strategico
func (sa *SnakeAgent) calculatePerimeterBonus() float64 {
	head := sa.game.GetSnake().GetHead()
	grid := sa.game.Grid

	// Controlla se la testa è sul perimetro
	isOnPerimeter := head.X == 0 || head.X == grid.Width-1 ||
		head.Y == 0 || head.Y == grid.Height-1

	if isOnPerimeter {
		// Bonus base per stare sul perimetro
		bonus := 5.0

		// Bonus aggiuntivo se il serpente è molto lungo (strategia efficace)
		if len(sa.game.GetSnake().Body) > 20 {
			bonus += 15.0
		}

		return bonus
	}

	return 0.0
}

// Reset prepara l'agente per una nuova partita mantenendo le conoscenze apprese.
func (sa *SnakeAgent) Reset() {
	// Preserve the existing stats
	existingStats := sa.game.Stats

	width := sa.game.Grid.Width
	height := sa.game.Grid.Height
	sa.game = NewGame(width, height)

	// Restore the stats
	sa.game.Stats = existingStats

	sa.agent.IncrementEpisode()
}

// SaveWeights salva i pesi della rete neurale su file.
func (sa *SnakeAgent) SaveWeights() error {
	return sa.agent.SaveWeights(qlearning.WeightsFile)
}
