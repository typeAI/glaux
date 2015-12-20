package glaux.persistence.mongodb

import glaux.interfaces.api.domain.{SessionId}
import glaux.linearalgebra.Tensor.TensorBuilder
import glaux.linearalgebra._
import Dimension._
import glaux.neuralnetwork.Net.DefaultNet
import glaux.neuralnetwork.layers._
import glaux.neuralnetwork._
import glaux.neuralnetwork.trainers.BatchTrainer.{LossInfo, BatchResult}
import glaux.neuralnetwork.trainers.SGD.SGDSettings
import glaux.reinforcementlearning.DeepMindQLearner.ConvolutionBased
import glaux.reinforcementlearning.Policy.AnnealingContext
import glaux.reinforcementlearning.QAgent.Session
import glaux.reinforcementlearning._
import glaux.reinforcementlearning.QLearner.{Transition, History, State, TemporalState}
import glaux.statistics.Probability
import play.api.libs.functional._
import syntax._
import reactivemongo.bson.Macros.Options.{AllImplementations, \/, UnionType}
import reactivemongo.bson._

import glaux.persistence.mongodb.GeneralHandlers._

object GlauxHandlers {


  val d1f = Macros.handler[Row]
  val d2f = Macros.handler[TwoD]
  val d3f = Macros.handler[ThreeD]
  val d4f = Macros.handler[FourD]

  implicit val gdimensionH: Handler[Dimension] = Macros.handlerOpts[Dimension, AllImplementations]

  case class InputDimensionHandler[T <: Tensor](dh: Handler[T#Dimensionality]) extends HandlerTrait[T#Dimensionality] {
    def read(bson: BSONDocument): T#Dimensionality = dh.read(bson)
    def write(t: T#Dimensionality): BSONDocument = dh.write(t)
  }

  implicit val rvih =  InputDimensionHandler[RowVector](d1f)
  implicit val mih =  InputDimensionHandler[Matrix](d2f)
  implicit val vih =  InputDimensionHandler[Vol](d3f)
  implicit val t4ih =  InputDimensionHandler[Tensor4](d4f)

  def tf[T <: Tensor : TensorBuilder: InputDimensionHandler]: Handler[T] = new HandlerTrait[T] {
    def read(bson: BSONDocument): T = ( bson.getAs[T#Dimensionality]("dimension").get,
      bson.getAs[Seq[Double]]("values").get)

    def write(t: T) = BSONDocument(
      "dimension" -> t.dimension.asInstanceOf[T#Dimensionality],
      "values" -> t.seqView
    )
  }

  implicit val rvf = tf[RowVector]
  implicit val mf = tf[Matrix]
  implicit val vf = tf[Vol]
  implicit val t3f = tf[Tensor4]

  implicit def inputLayerF[Input <: Tensor: InputDimensionHandler] = {
    field[Input#Dimensionality]("dimension").inmap(InputLayer.apply[Input], unlift(InputLayer.unapply[Input]))
  }

  implicit val rsf = Macros.handler[RegularizationSetting]
  implicit val rectanglef = Macros.handler[Rectangle]
  implicit val poolF = Macros.handler[Pool]
  implicit val convH = Macros.handler[Convolution]

  implicit def fcf[T <: Tensor : TensorBuilder : InputDimensionHandler] = (
     field[Matrix]("filter") ~
     field[RowVector]("bias") ~
     field[RegularizationSetting]("filterRegularization") ~
     field[Dimension]("inDimension").cast[T#Dimensionality] ~
     field[String]("id"))(FullyConnected.apply[T], unlift(FullyConnected.unapply[T]))

  implicit def reluf[T <: Tensor : TensorBuilder : InputDimensionHandler] =
    (field[Dimension]("dimension").cast[T#Dimensionality] //todo: try field[T#Dimensionality]
      ~ field[String]("id"))(Relu.apply[T], unlift(Relu.unapply[T]))

  implicit val hiddenLayerH = polymorphic[HiddenLayer]( new Partial[HiddenLayer, Convolution],
                                                        new Partial[HiddenLayer, Pool],
                                                        new GPartial[HiddenLayer, FullyConnected[Matrix]]({ case FullyConnected(_, _, _, TwoD(_, _), _) => }),
                                                        new GPartial[HiddenLayer, FullyConnected[RowVector]]({ case FullyConnected(_, _, _, Row(_), _) => }),
                                                        new GPartial[HiddenLayer, FullyConnected[Vol]]({ case FullyConnected(_, _, _, ThreeD(_, _, _), _) => }),
                                                        new GPartial[HiddenLayer, Relu[RowVector]]({ case Relu(Row(_), _) => }),
                                                        new GPartial[HiddenLayer, Relu[Vol]]({ case Relu(ThreeD(_, _, _),_) => }),
                                                        new GPartial[HiddenLayer, Relu[Matrix]]({case l @ Relu(TwoD(_, _),_) => }) )


  implicit val rlh = Macros.handler[Regression]
  implicit val smxh = Macros.handler[Softmax]
  implicit val lossh = Macros.handler[LossInfo]

  implicit val lossLayerH = polymorphic[LossLayer]( new Partial[LossLayer, Regression],
                                                    new Partial[LossLayer, Softmax])

  implicit def dftNetH[Input <: Tensor : InputDimensionHandler] = (
      field[InputLayer[Input]]("inputLayer") ~
      field[Seq[HiddenLayer]]("hiddenLayers") ~
      field[LossLayer]("lossLayer")
    )(DefaultNet.apply[Input], unlift(DefaultNet.unapply[Input]))


  implicit def dbrh[Input <: Tensor : InputDimensionHandler] = (
      field[LossInfo]("lossInfo") ~
        field[DefaultNet[Input]]("net") ~
        field[Int]("batchSize") ~
        field[Unit]("calculationContext")
    )(BatchResult.apply[DefaultNet[Input], Unit], unlift(BatchResult.unapply[DefaultNet[Input], Unit]))


  implicit def temporalStateH[Input <: Tensor : TensorBuilder : Handler] =
    (field[Input]("readings") ~
     field[Time]("time")
    )(TemporalState.apply[Input], unlift(TemporalState.unapply[Input]))

  implicit def qStateH[Input <: Tensor : TensorBuilder : Handler] =
    (field[Seq[TemporalState[Input]]]("fullHistory") ~
    field[Boolean]("isTerminal"))(State.apply[Input], unlift(State.unapply[Input]))
  
  implicit def transitionH[Input <: Tensor : TensorBuilder : Handler] =
    ( field[State[Input]]("before") ~
      field[Action]("action") ~
      field[Reward]("reward") ~
      field[State[Input]]("after"))(Transition.apply[Input], unlift(Transition.unapply[Input]))


  implicit def learnerIterationH[T <: DeepMindQLearner](learner: T)(
    implicit nh: Handler[learner.Net], trh: Handler[learner.TrainingResult], transitionH: Handler[Transition[learner.Input]], sh: Handler[learner.State]) = {
    //manual method to function conversion is needed because scala compiler can't convert method with dependent type
    import learner.{Net, Memory, TrainingResult, State}
    val apply = (n: Net, m: Memory, r: TrainingResult, s: State, it: Boolean, tnhc: Int) => learner.DeepMindIteration(n, m, r, s, it, tnhc)
    val unapply = (i: learner.DeepMindIteration) => (i.net, i.memory, i.trainingResult, i.state, i.isTerminal, i.targetNetHitCount)

    ( field[Net]("targetNet") ~
      field[Memory]("memory") ~
      field[TrainingResult]("trainingResult") ~
      field[State]("state") ~
      field[Boolean]("isTerminal") ~
      field[Int]("targetNetHitCount"))(apply, unapply)
  }

  implicit val probabilityR: Reader[Probability] = new BSONReader[BSONDouble, Probability] {
    def read(bson: BSONDouble): Probability = Probability(bson.value)
  }

  implicit val probabilityW: Writer[Probability] = new BSONWriter[Probability, BSONDouble] {
    def write(p: Probability): BSONDouble = BSONDouble(p.value)
  }

  implicit val ach =  Macros.handler[AnnealingContext]

  implicit def agentSessionH(implicit agent: SimpleQAgent): Handler[agent.Session] = {
    implicit val iH = learnerIterationH(agent.qLearner)
    sessionH[agent.qLearner.Iteration, agent.policy.Context]
  }

  implicit def agentSessionH(implicit agent: AdvancedQAgent): Handler[agent.Session] = {
    implicit val iH = learnerIterationH(agent.qLearner)
    sessionH[agent.qLearner.Iteration, agent.policy.Context]
  }

  def sessionH[ Iteration <: QLearner#IterationLike: Handler,
                Context <: Policy.DecisionContext : Handler]: Handler[Session[Iteration, Context]] = {
    val apply = (i: Iteration, cr: Reward, crd: Vector[Reading], dc: Context, la: Option[Action], ic: Boolean) => Session(i, cr, crd, dc, la, ic)

    val unapply = (s: Session[Iteration, Context]) => (s.iteration, s.currentReward, s.currentReadings, s.decisionContext, s.lastAction, s.isClosed)

    ( field[Iteration]("iteration") ~
      field[Reward]("currentReward") ~
      field[Vector[Reading]]("currentReadings") ~
      field[Context]("decisionContext") ~
      nullableField[Action]("lastAction") ~
      field[Boolean]("isClosed"))(apply, unapply)
  }
  import ConvolutionBased.Settings
  implicit val cblsh = Macros.handler[Settings]
  implicit val sgdlsh = Macros.handler[SGDSettings]

}
