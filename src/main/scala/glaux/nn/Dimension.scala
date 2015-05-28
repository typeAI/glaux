package glaux.nn

import org.nd4j.linalg.api.ndarray.INDArray

trait Dimension

object Dimension {
  type Shape = Array[Int]

  trait DimensionFactory[DT <: Dimension] {
    val createFunction: PartialFunction[Shape, DT]
    lazy val tryCreate: Shape => Option[DT] = createFunction.lift
    def create(shape: Shape): DT = tryCreate(shape).getOrElse(throw unsupported(shape))
  }

  def unsupported(shape: Shape):Exception = new UnsupportedOperationException(s"unrecognized matrix shape $shape")

  def ofSpecific[DT <: Dimension](matrix: INDArray)(implicit df: DimensionFactory[DT]): DT = {
    val shape = matrix.shape()
    df.tryCreate(shape).getOrElse(throw unsupported(shape))
  }

  def of(matrix: INDArray): Dimension = {
    val shape = matrix.shape()
    val df3d = implicitly[DimensionFactory[ThreeD]]
    val df1d = implicitly[DimensionFactory[Row]]
    df1d.tryCreate(shape).orElse(df3d.tryCreate(shape)).getOrElse(throw unsupported(shape))
  }

  case class ThreeD(x: Int, y: Int, z: Int) extends Dimension {
    assert(x > 0 && y > 0 && z > 0)
  }

  case class Row(size: Int) extends Dimension {
    assert(size > 0)
  }

  implicit object Row extends DimensionFactory[Row] {
    val dimIndexOfData: Int = 1

    val createFunction: PartialFunction[Shape, Row] = {
      case Array(1, size) => Row(size)
    }
  }

  implicit object ThreeD extends DimensionFactory[ThreeD] {
    val createFunction = PartialFunction[Shape, ThreeD] {
      case Array(x, y, z) => ThreeD(x,y,z)
    }
  }

}
