package glaux.nn

trait Trainer {
  type Result <: IterationResult
  type IterationContext = Result#MethodSpecificContext

  def init(net: Net): Result
  def iterate(input: Input, target: Output, lastIterationResult: Result): Result = {
    val net = lastIterationResult.net
    val dataFlow = net.forward(input)
    val (loss, paramsGrads) = net.backward(target, dataFlow)

    val (newParams, newContext) = performUpdate(paramsGrads, lastIterationResult)

    val newLayers = newParams.groupBy(_.layer).map {
      case (l, ps) => l.updateParams(ps)
    }
    createResult(loss, net.updateLayers(newLayers), lastIterationResult.iterationNum + 1, newContext)
  }

  def createResult(loss: Loss, newNet: Net, iterationNum: Int, newContext: IterationContext): Result

  def performUpdate(paramGrads: Seq[ParamGradient], lastIterationResult: Result): (Seq[LayerParam], IterationContext)
}


trait IterationResult {
  def lossInfo: LossInfo
  def net: Net
  def iterationNum: Int
  type MethodSpecificContext

}

case class LossInfo(l1Decay: Loss, l2Decay: Loss, costLoss: Loss, loss: Loss)