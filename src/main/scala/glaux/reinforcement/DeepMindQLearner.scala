package glaux.reinforcement

import glaux.linalg.Dimension.Row
import glaux.linalg.{Dimension, RowVector, Tensor}
import glaux.nn.trainers.BatchTrainer
import glaux.reinforcement.QLearner._

import scala.util.Random

/**
 * QLearner based on deepmind algorithm
 */
trait DeepMindQLearner extends QLearner {
  import MyStateTypes._
  val historyLength: Int
  val inputDimension: Input#Dimensionality
  val gamma: Int
  val batchSize: Int
  val targetNetUpdateFreq: Int //avg # of iterations before updating the target net

  val minMemorySizeBeforeTraining: Int

  protected def validate: Unit = {
    assert(minMemorySizeBeforeTraining > batchSize, "must have enough transitions in memory before training")
  }

  def iterate(lastIteration: Iteration, observation: Observation): Iteration = {

    assert(observation.recentHistory.forall(_.readings.dimension == inputDimension), "input readings doesn't conform to preset reading dimension")

    def updateMemory: Memory = {
      val before = lastIteration.memory.last.after

      val relevantPreviousHistory = before.fullHistory.filter(_.time.isBefore(observation.recentHistory.head.time))
      val after = State((relevantPreviousHistory ++ observation.recentHistory).takeRight(historyLength), observation.isTerminal)

      val action = observation.lastAction
      lastIteration.memory :+ Transition(before, action, observation.reward, after)
    }

    val newMemory = updateMemory
    val doTraining: Boolean = newMemory.size > minMemorySizeBeforeTraining
    val updateTarget: Boolean = doTraining && lastIteration.targetNetHitCount > targetNetUpdateFreq

    val targetNet = lastIteration.targetNet
    lazy val newResult = train(newMemory, lastIteration.trainingResult, targetNet)

    Iteration(
      if(updateTarget) newResult.net else targetNet,
      newMemory,
      if(doTraining) newResult else lastIteration.trainingResult,
      if(updateTarget) 0 else lastIteration.targetNetHitCount + 1
    )

  }

  private def train(memory: Memory, lastResult: TrainerResult, targetNet: Net): TrainerResult = {
    def randomExamples: Memory = {
      (1 to batchSize).map { _ =>
        memory(Random.nextInt(memory.size))
      }
    }

    def toTrainingInput(transition: Transition): (NetInput, Trainer#ScalarOutputInfo) = {
      val regressionOnAction = if (transition.after.isTerminal) transition.reward else
        transition.reward + targetNet.predict(transition.after).seqView.max * gamma

      (transition.before, (regressionOnAction, transition.action))
    }

    trainer.trainBatchWithScalaOutputInfo(randomExamples.map(toTrainingInput), lastResult)
  }

}

object DeepMindQLearner {
  case class Simplified(  historyLength: Int = 50,
                          inputDimension: Row,
                          gamma: Int,
                          batchSize: Int = 32,
                          targetNetUpdateFreq: Int = 10, //avg # of iterations before updating the target net
                          override protected val trainer: Simplified#Trainer,
                          minMemorySizeBeforeTraining: Int = 100 ) extends DeepMindQLearner {
    type NetInput = RowVector
    type Input = RowVector

    import MyStateTypes._

    validate

    implicit def toInput(state: State): NetOutput = {
      RowVector(inputDimension,  state.fullHistory.flatMap(_.readings.seqView))
    }

    def buildNet: Net = ???
  }


}