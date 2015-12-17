package glaux.reinforcementlearning

import glaux.linearalgebra.RowVector
import glaux.reinforcementlearning.Policy.{ AnnealingContext, Annealing }
import glaux.reinforcementlearning.QLearner.State
import org.specs2.mutable.Specification
import glaux.statistics.Probability

class AnnealingPolicySpec extends Specification {
  val policy = Annealing[State[RowVector]](4, 0.5, 10000)
  import policy.QFunction

  val aState: State[RowVector] = State(Nil, false)
  val aQFunc: QFunction = (_, _) ⇒ 0

  "reduce exploreProbability until no steps left for exploration" >> {
    val stepsLeft = 10
    val initContext = AnnealingContext(policy.minExploreProbability + 0.3, stepsLeft)

    val lastContext = (1 until stepsLeft).foldLeft(initContext) { (ctx, _) ⇒
      val (_, resultContext) = policy.decide(aState, aQFunc, ctx)
      resultContext.explorationProbability must be_>(policy.minExploreProbability)
      resultContext.explorationProbability must be_<(ctx.explorationProbability)
      resultContext.stepsLeftForExploration must be_<(ctx.stepsLeftForExploration)
      resultContext
    }

    lastContext.stepsLeftForExploration === 1
    val (_, finalContext) = policy.decide(aState, aQFunc, lastContext)
    finalContext.explorationProbability === policy.minExploreProbability

  }

  "keep at min exploration Rate after exploration period" >> {
    val lastCtx = AnnealingContext(policy.minExploreProbability, 0)
    val (_, result) = policy.decide(aState, aQFunc, lastCtx)
    result.explorationProbability === policy.minExploreProbability
  }

  "pick the action with Max Q when exploration rate is zero" >> {
    val policy = Annealing[State[RowVector]](4, 0d, 10000)
    val lastCtx = AnnealingContext(0d, 0)
    val pick3: QFunction = (_, a) ⇒ if (a == 3) 10 else 2

    val actions = (1 to 100).map(_ ⇒ policy.decide(aState, pick3, lastCtx)._1).distinct
    actions.length === 1
    actions.head === 3
  }

  "pick random action during exploratio time" >> {

    val initContext = AnnealingContext(1d, 100000)

    val actions = (1 to 100).scanLeft((0, initContext)) { (p, _) ⇒
      policy.decide(aState, aQFunc, p._2)
    }.map(_._1).distinct

    actions.length === 4 //all four actions are used

  }

}
