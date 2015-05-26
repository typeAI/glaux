package glaux.nn

trait Trainer {
  type Input <: Vol
  type Trainee <: Net[Input]
  type Result <: IterationResult[Trainee]
  type IterationContext = Result#MethodSpecificContext
  
  implicit val updater: Updater[Trainee]
  
  def init(net: Trainee): Result


  def iterate(input: Input, target: Trainee#Output, lastResult: Result): Result = {
    val net: Trainee = lastResult.net
    val dataFlow = net.forward(input)
    val (loss, paramsGrads) = net.backward(target, dataFlow)

    val (newParams, newContext) = performUpdate(paramsGrads, lastResult)

    val newLayers = newParams.groupBy(_.layer).map {
      case (l, ps) => l.updateParams(ps)
    }
    createResult(loss, updater.update(net, newLayers), lastResult.iterationNum + 1, newContext)
  }

  protected def createResult(loss: Loss, newNet: Trainee, iterationNum: Int, newContext: IterationContext): Result

  def performUpdate(paramGrads: Seq[ParamGradient], lastIterationResult: Result): (Seq[LayerParam], IterationContext)
}


trait IterationResult[NT <: Net[_]] {
  def lossInfo: LossInfo
  def net: NT
  def iterationNum: Int
  type MethodSpecificContext
}

case class LossInfo(l1Decay: Loss, l2Decay: Loss, costLoss: Loss, loss: Loss)