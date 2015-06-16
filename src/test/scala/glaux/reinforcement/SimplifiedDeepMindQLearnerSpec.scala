package glaux
package reinforcement

import java.time.{Clock, LocalTime, LocalDate, ZonedDateTime}

import glaux.linalg.Dimension.Row
import glaux.linalg.RowVector

import org.specs2.mutable.Specification

class SimplifiedDeepMindQLearnerSpec extends Specification {
  val start = ZonedDateTime.of(LocalDate.of(2015, 2, 14), LocalTime.of(14, 30), Clock.systemDefaultZone().getZone)
  val learner = DeepMindQLearner.Simplified(historyLength = 2)

  import learner.{Observation, TemporalState} //todo: can this import be easier (don't need to remember)

  val init = learner.init(Row(2),2)

  "init" >> {

    "memory" >> {
      init.memory must beEmpty
    }

    "net" >> {
      init.net.inputDimension must_==(Row(4))
      init.net.outputDimension must_==(Row(2))
    }

    "trainResult" >> {
      init.trainingResult.batchSize === 0
    }

  }

  "initial iterations" >> {

    val firstIter = learner.iterate(init, Observation(lastAction = 1,
                                                    reward = 0,
                                                    recentHistory = Seq(TemporalState(RowVector(1, 0), start)),
                                                    isTerminal = false
                                                  ))
    "first iter" >> {
      "memory without enough history" >> {
        firstIter.memory must beEmpty
      }

      "recentHistory" >> {
        firstIter.recentHistory.head.readings must_== RowVector(1, 0)
      }
    }

    val secondIter = learner.iterate(firstIter, Observation(lastAction = 0,
      reward = 0,
      recentHistory = Seq(TemporalState(RowVector(2, 1), start.plusMinutes(1))),
      isTerminal = false))

    "second iter" >> {
      "still not enough history for memory" >> {
        secondIter.memory must beEmpty
      }

      "building recentHistory" >> {
        secondIter.recentHistory.size === 2
      }

    }

    val thirdIter = learner.iterate(secondIter, Observation(lastAction = 1,
      reward = 1,
      recentHistory = Seq(TemporalState(RowVector(3, 2), start.plusMinutes(2))),
      isTerminal = true))

    "third iter" >> {
      "has enough history for memory" >> {
        thirdIter.memory.size === 1
      }

      "build the trainsition" >> {
        val transition = thirdIter.memory.head
        transition.before.fullHistory.map(_.readings) must_== Seq(RowVector(1,0), RowVector(2,1))
        transition.after.fullHistory.map(_.readings) must_== Seq(RowVector(2,1), RowVector(3,2))
        transition.action === 1
        transition.reward === 1
      }

      "keep only relevant recentHistory" >> {
        thirdIter.recentHistory.size === 2
      }

      "latest state is terminal" >> {
        thirdIter.latestState.map(_.isTerminal) must beSome(true)
      }

    }

    val fouth = learner.iterate(thirdIter, Observation(lastAction = 1,
      reward = 1,
      recentHistory = Seq(TemporalState(RowVector(4, 3), start.plusMinutes(3)),
                          TemporalState(RowVector(4, 3), start.plusMinutes(4))),
      isTerminal = false))

    "fourth iter" >> {
      "add to memory because last one is terminated" >> {
        fouth.memory.size === 1
      }
    }

    val fifth = learner.iterate(fouth, Observation(lastAction = 3,
      reward = 2,
      recentHistory = Seq(TemporalState(RowVector(5, 4), start.plusMinutes(5)),
                          TemporalState(RowVector(5, 4), start.plusMinutes(6))),
      isTerminal = false))

    "fifth iter" >> {
      "build new memory" >> {
        fifth.memory.size === 2
      }

      "build the transition" >> {
        val transition = fifth.memory.last
        transition.before.fullHistory.map(_.readings) must_== Seq(RowVector(4,3), RowVector(4,3))
        transition.after.fullHistory.map(_.readings) must_== Seq(RowVector(5,4), RowVector(5,4))
        transition.action === 3
        transition.reward === 2
      }

    }

  }

}
