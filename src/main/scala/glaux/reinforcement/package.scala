package glaux

import java.time.ZonedDateTime

import glaux.reinforcement.QLearner.{TemporalState, State}

package object reinforcement {

  type Action = Int

  type Reward = Double

  type Q = Double

  type Time = ZonedDateTime

}
