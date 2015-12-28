package glaux.reinforcementlearning

import java.time.{Clock, LocalTime, LocalDate, ZonedDateTime}

import glaux.linearalgebra.Dimension.Row
import glaux.linearalgebra.RowVector
import glaux.reinforcementlearning.DeepMindQLearner.ConvolutionBased
import glaux.reinforcementlearning.QLearner._
import org.specs2.mutable.Specification

import scala.util.Random

class ConvolutionBasedSpec extends Specification {
  val start = ZonedDateTime.of(2015, 2, 14, 14, 30, 0, 0, Clock.systemDefaultZone().getZone)
  val learner = ConvolutionBased(settings = ConvolutionBased.Settings(historyLength = 10, filterSize = 3, batchSize = 3, minMemorySizeBeforeTraining = 10))
  val numOfAction = 3
  def mockHistory(from: ZonedDateTime, length: Int = 10, numOfReadings: Int = 5) = {
    def randomReading = RowVector((0 until numOfReadings).map(_ ⇒ Random.nextDouble()): _*)

    (0 until length).map { i ⇒
      TemporalState(randomReading, from.plusMinutes(i))
    }
  }

  def mockObservation(minuteFromStart: Int) = Observation(
    lastAction = Random.nextInt(numOfAction),
    reward = if (Random.nextBoolean) Random.nextInt(5) else 0,
    recentHistory = mockHistory(start.plusMinutes(minuteFromStart)),
    isTerminal = false
  )

  "init" >> {
    val initIter = learner.init(mockHistory(start), numOfAction)
    initIter.net.outputDimension === Row(numOfAction)
  }

  "training" >> {
    val init = learner.init(mockHistory(start), numOfAction)

    "one iteration" >> {
      val iter = learner.iterate(init, Observation(
        lastAction = 0,
        reward = 1,
        recentHistory = mockHistory(start.plusMinutes(20)),
        isTerminal = false
      ))
      iter.memory.size === 1
      init.trainingResult.batchSize === 0
    }

    "start training after minMemorySizeBeforeTraining of meaningful iteration" >> {
      val result = (0 to (learner.minMemorySizeBeforeTraining + 1)).foldLeft(init) { (iter, i) ⇒
        learner.iterate(iter, mockObservation(i * learner.historyLength + 1))
      }
      result.trainingResult.batchSize === learner.batchSize
    }

  }

}
