package glaux.reinforcement.integration

import java.time.{Clock, LocalTime, LocalDate, ZonedDateTime}

import glaux.linalg.RowVector
import glaux.nn.trainers.VanillaSGD
import glaux.nn.trainers.SGD.SGDSettings
import glaux.reinforcement.DeepMindQLearner.Simplified
import glaux.reinforcement.QLearner._
import glaux.reinforcement.{Time, Action, DeepMindQLearner}
import org.specs2.mutable.Specification

import scala.util.Random

class SimplifiedIntegration extends Specification {

  val start = ZonedDateTime.of(LocalDate.of(2015, 2, 14), LocalTime.of(14, 30), Clock.systemDefaultZone().getZone)
//  val trainer = MomentumSGD[Simplified#Net](MomentumSGDOptions(SGDOptions(learningRate = 0.005), momentum = 0.9))
  val trainer = VanillaSGD[Simplified#Net](SGDSettings(learningRate = 0.05))
  val learner = DeepMindQLearner.Simplified(historyLength = 2, batchSize = 20, trainer = trainer)
  import learner.{State, History}
  

  def randomBinary = if(Random.nextBoolean) 1 else 0
  def randomReading = RowVector(randomBinary, randomBinary, randomBinary)

  def randomHistory(from: Time): History = Seq(
    TemporalState(randomReading, from),
    TemporalState(randomReading, from.plusMinutes(1)),
    TemporalState(randomReading, from.plusMinutes(2)))
  def randomTerminal: Boolean = Random.nextDouble > 0.97

  def newObservation(lastState: State, lastAction: Action): learner.Observation = {
    val time = lastState.endTime.plusMinutes(1)
    val reward = {
      if(lastState.isTerminal) 0 else {
        //reward when action matches the reading, that is, sum of three readings in the index exceed certain threshed
        if(lastState.fullHistory.takeRight(2).map(_.readings(lastAction)).sum > 1.5 ) 1.0 else 0
      }
    }
    Observation(lastAction, reward, randomHistory(time), randomTerminal)
  }
  
  val init = learner.init(randomHistory(start), 3)
  
  "can learn the right action" >> {
    //learning
    val lastIter = (1 to 500).foldLeft(init) { (lastIteration, _) =>
      val obs = newObservation(lastIteration.state, Random.nextInt(3))
      learner.iterate(lastIteration, obs)
    }

    val testSize = 100
    //testing
    val results = (1 to testSize).scanLeft(lastIter) { (lastIteration, _) =>
      val result = learner.iterate(lastIteration, newObservation(lastIteration.state, Random.nextInt(3)))

      result
    }.filterNot(_.actionQs.isEmpty)

    val correct = results.filter { result =>
      val cumReading = result.state.fullHistory.map(_.readings).takeRight(2).reduce(_ + _).seqView
      val correctActions = cumReading.zipWithIndex.filter(_._1 > 1.5).map(_._2)
      val predictedAction = result.actionQs.maxBy(_._2)._1
//      println(cumReading.map( v => (v * 10).toInt))
//      println(result.actionQs.mapValues(v => (v * 100).toInt ))
      correctActions.contains(predictedAction) || correctActions.isEmpty
    }
    val correctionRate = correct.size.toDouble / results.size
//    println("correction rate " + correctionRate)
    correctionRate must be_>=(0.60)
  }

}
