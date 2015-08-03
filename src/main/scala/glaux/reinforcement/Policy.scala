package glaux.reinforcement

import glaux.reinforcement.Policy.{AnnealingContext, DecisionContext}
import glaux.reinforcement.QLearner.State
import glaux.statistics.Probability

import scala.util.Random

trait Policy[StateT <: State[_]] {
  type Context <: DecisionContext
  type QFunction = (StateT, Action) => Q
  def numOfActions: Int

  def decide(state: StateT, qFunction: QFunction, context: Context): (Action, Context)

  def init: Context

}


object Policy {
  trait DecisionContext
  type NumberOfSteps = Int

  case class Annealing[StateT <: State[_]]( numOfActions: Action,
                                            minExploreProbability: Probability,
                                            lengthOfExploration: NumberOfSteps ) extends Policy[StateT] {
    type Context = AnnealingContext
    def decide(state: StateT, qFunction: QFunction, context: Context): (Action, Context) = {

      def actionWithMaxQ = (0 until numOfActions).map(qFunction(state, _)).zipWithIndex.maxBy(_._1)._2

      val explorationProbability = if(context.explorationProbability > minExploreProbability) {
        context.explorationProbability - ((context.explorationProbability - minExploreProbability) / context.stepsLeftForExploration)
      } else minExploreProbability
      val action = if(explorationProbability.nextBoolean())
        Random.nextInt(numOfActions)
      else
        actionWithMaxQ

      (action, AnnealingContext(explorationProbability, Math.max(context.stepsLeftForExploration - 1, 0)))
    }

    def init: AnnealingContext = AnnealingContext(Probability(1), lengthOfExploration)
  }

  case class AnnealingContext(explorationProbability: Probability,
                              stepsLeftForExploration: NumberOfSteps) extends DecisionContext


}
