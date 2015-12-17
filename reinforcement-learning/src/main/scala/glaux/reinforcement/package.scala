package glaux

import java.time.ZonedDateTime

package object reinforcementlearning {

  type Action = Int

  type Reward = Double

  type Q = Double

  type Time = ZonedDateTime

  type Reading = (Seq[Double], Time)

}
