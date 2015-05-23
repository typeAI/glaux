package glaux.nn

trait Trainer {
  def net: Net
  def train(input: Input, target: Output): LossInfo = {
    val actual = net.forward(input, true)
    val (loss, paramsGrads) = net.backward(target, actual)
    val newParams = paramsGrads.map(newParam)
    newParams.groupBy(_.layer).foreach {
      case (l, ps) => l.updateParams(ps)
    }
    ???
  }

  def newParam(paramGradient: ParamGradient): Param
}

case class LossInfo(l1Decay: Loss, l2Decay: Loss, costLoss: Loss, loss: Loss)