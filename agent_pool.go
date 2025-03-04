package main

import (
	"snake-game/qlearning"
	"sync"
)

// AgentPool gestisce un pool di agenti che condividono la stessa rete neurale
type AgentPool struct {
	agents        []*SnakeAgent
	sharedNetwork *qlearning.Agent
	mutex         sync.RWMutex
}

// NewAgentPool crea un nuovo pool di agenti
func NewAgentPool(numAgents int, gameWidth, gameHeight int) *AgentPool {
	// Crea un singolo network condiviso
	sharedNetwork := qlearning.NewAgent(0.001, 0.99)

	pool := &AgentPool{
		agents:        make([]*SnakeAgent, numAgents),
		sharedNetwork: sharedNetwork,
	}

	// Inizializza gli agenti
	for i := 0; i < numAgents; i++ {
		game := NewGame(gameWidth, gameHeight)
		pool.agents[i] = NewSnakeAgent(game)
		// Sostituisce il network dell'agente con quello condiviso
		pool.agents[i].agent = sharedNetwork
	}

	return pool
}

// GetAgent restituisce un agente specifico dal pool
func (p *AgentPool) GetAgent(index int) *SnakeAgent {
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	if index >= 0 && index < len(p.agents) {
		return p.agents[index]
	}
	return nil
}

// GetAllAgents restituisce tutti gli agenti nel pool
func (p *AgentPool) GetAllAgents() []*SnakeAgent {
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	return p.agents
}

// SaveWeights salva i pesi della rete neurale condivisa
func (p *AgentPool) SaveWeights() error {
	return p.sharedNetwork.SaveWeights(qlearning.WeightsFile)
}

// Cleanup rilascia le risorse utilizzate dal pool
func (p *AgentPool) Cleanup() {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	// Cleanup degli agenti
	for _, agent := range p.agents {
		if agent != nil {
			agent.game = nil // Rimuovi riferimento al gioco
		}
	}

	// Cleanup del network condiviso
	if p.sharedNetwork != nil {
		p.sharedNetwork.Cleanup()
		p.sharedNetwork = nil
	}

	p.agents = nil
}
