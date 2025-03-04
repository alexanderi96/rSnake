package main

// abs restituisce il valore assoluto di un intero.
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// minInt32 restituisce il valore minimo tra due int32
func minInt32(a, b int32) int32 {
	if a < b {
		return a
	}
	return b
}

// manhattanDistance calcola la distanza di Manhattan tra due punti senza wrapping
func manhattanDistance(p1, p2 Point, width, height int) int {
	return abs(p2.X-p1.X) + abs(p2.Y-p1.Y)
}
