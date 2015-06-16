package glaux.reinforcement.integration

import java.time.{Clock, LocalTime, LocalDate, ZonedDateTime}

import glaux.linalg.Dimension.Row
import glaux.linalg.RowVector
import glaux.reinforcement.{Time, Action, DeepMindQLearner}
import org.specs2.mutable.Specification

import scala.util.Random

class SimplifiedIntegration extends Specification {

  val start = ZonedDateTime.of(LocalDate.of(2015, 2, 14), LocalTime.of(14, 30), Clock.systemDefaultZone().getZone)
  val learner = DeepMindQLearner.Simplified(historyLength = 3)
  import learner.{Observation, TemporalState, State, History}
  val init = learner.init(Row(3),3)

  def randomReading = RowVector(Random.nextDouble, Random.nextDouble, Random.nextDouble)

  def randomHistory(start: Time): History = Seq(
    TemporalState(randomReading, start),
    TemporalState(randomReading, start.plusMinutes(1)),
    TemporalState(randomReading, start.plusMinutes(2)))
  def randomTerminal: Boolean = Random.nextDouble > 0.97

  def newObservation(lastState: Option[State], lastAction: Action): Observation = {
    val time = lastState.map(_.endTime.plusMinutes(1)).getOrElse(ZonedDateTime.now)
    val reward = {
      if(lastState.map(_.isTerminal).getOrElse(true)) 0 else {
        //reward when action matches the reading, that is, sum of three readings in the index exceed certain threshed
        if(lastState.get.fullHistory.takeRight(2).map(_.readings(lastAction)).sum > (Random.nextDouble() * 2 + 0.5) ) 1.0 else 0
      }
    }
    Observation(lastAction, reward, randomHistory(time), randomTerminal)
  }

  "can learn the right action" >> skipped {
    //learning
    val lastIter = (1 to 20000).foldLeft(init) { (lastIteration, _) =>
      print(".")
      learner.iterate(lastIteration, newObservation(lastIteration.latestState, Random.nextInt(3)))

    }

    val testSize = 100
    //testing
    val results = (1 to testSize).scanLeft(lastIter) { (lastIteration, _) =>
      val result = learner.iterate(lastIteration, newObservation(lastIteration.latestState, Random.nextInt(3)))

      result
    }.filterNot(_.actionQs.isEmpty)

    val correct = results.filter { result =>

      val cumReading = result.latestState.get.fullHistory.map(_.readings).takeRight(2).reduce(_ + _).seqView
      val correctAction = cumReading.zipWithIndex.maxBy(_._1)._2
      val predictedAction = result.actionQs.maxBy(_._2)._1
//      println(cumReading.map( v => (v * 10).toInt))
//      println(result.actionQs.mapValues(v => (v * 100).toInt ))
      correctAction == predictedAction
    }
    val correctionRate = correct.size.toDouble / results.size
    println("correction rate " + correctionRate)
    correctionRate must be_>=(0.7)
  }

}
