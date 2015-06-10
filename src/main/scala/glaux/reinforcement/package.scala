package glaux

import glaux.linalg.RowVector
import glaux.reinforcement.QLearner.State

package object reinforcement {
  type TemporalState = RowVector

  type History = Seq[TemporalState]
  type Action = Int

  type Reward = Double

  type Q = Double

  type Policy = (State, Action => Q) => Action
}
