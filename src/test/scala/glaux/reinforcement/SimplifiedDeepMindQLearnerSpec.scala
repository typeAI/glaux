package glaux.reinforcement

import glaux.linalg.Dimension.Row
import glaux.nn.trainers.{SGDOptions, VanillaSGD}
import org.specs2.mutable.Specification

class SimplifiedDeepMindQLearnerSpec extends Specification {

  val trainer = VanillaSGD[DeepMindQLearner.Simplified#Net](SGDOptions())
  val learner = DeepMindQLearner.Simplified(trainer)

  "init" >> {
    "update memory" >> {
      val init = learner.init(Row(2),2)
      init.memory must beEmpty
    }
  }
}
