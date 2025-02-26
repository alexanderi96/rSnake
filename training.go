package main

import (
	"fmt"
)

// Train esegue l'addestramento dell'agente
func Train(totalEpisodes int, gridWidth, gridHeight int) {
	game := NewGame(gridWidth, gridHeight)
	agent := NewSnakeAgent(game)

	// Statistiche
	bestScore := 0
	totalScore := 0

	for episode := 0; episode < totalEpisodes; episode++ {
		// Reset del gioco per il nuovo episodio
		agent.Reset()

		// Esegui l'episodio
		for !game.GetSnake().Dead {
			agent.Update()
		}

		// Aggiorna le statistiche
		score := game.GetSnake().Score
		totalScore += score
		if score > bestScore {
			bestScore = score
		}

		// Reset statistics for next batch
		if (episode+1)%50 == 0 {
			totalScore = 0
		}

		// Save weights periodically
		if (episode+1)%500 == 0 {
			if err := agent.SaveWeights(); err != nil {
				fmt.Printf("Error saving weights: %v\n", err)
			}
		}
	}
}
