package glaux.linalg

import Dimension.Shape

sealed trait Dimension {
  lazy val totalSize: Int = shape.reduce(_ * _)
  def shape: Shape
}

object Dimension {
  type Shape = Array[Int]

  trait DimensionFactory[DT <: Dimension] {
    val createFunction: PartialFunction[Shape, DT]
    lazy val tryCreate: Shape => Option[DT] = createFunction.lift
    def create(shape: Shape): DT = tryCreate(shape).getOrElse(throw unsupported(shape))
  }

  private def unsupported(shape: Shape):Exception = new UnsupportedOperationException(s"unrecognized INDArray shape (${shape.mkString(",")})")

  def ofSpecific[DT <: Dimension](shape: Shape)(implicit df: DimensionFactory[DT]): DT = {
    df.tryCreate(shape).getOrElse(throw unsupported(shape))
  }

  def of(shape: Shape): Dimension = {
    val df4d = implicitly[DimensionFactory[FourD]]
    val df3d = implicitly[DimensionFactory[ThreeD]]
    val df2d = implicitly[DimensionFactory[TwoD]]
    val df1d = implicitly[DimensionFactory[Row]]
    (Seq(df4d, df3d, df2d, df1d).foldLeft[Option[Dimension]](None) { (lr, f) => lr orElse f.tryCreate(shape) }).getOrElse(throw unsupported(shape))
  }

  case class FourD(x: Int, y: Int, z: Int, f: Int) extends Dimension {
    assert(x > 0 && y > 0 && z > 0 && f > 0)
    def shape: Shape = Array(x, y, z, f)
  }

  case class ThreeD(x: Int, y: Int, z: Int) extends Dimension {
    assert(x > 0 && y > 0 && z > 0)
    def shape: Shape = Array(x, y, z)
  }

  case class TwoD(x: Int, y: Int) extends Dimension {
    assert(x > 0 && y > 0)
    def shape: Shape = Array(x, y)
  }

  case class Row(size: Int) extends Dimension {
    assert(size > 0)
    def shape: Shape = Array(1, size)
  }

  implicit object Row extends DimensionFactory[Row] {
    val dimIndexOfData: Int = 1

    val createFunction: PartialFunction[Shape, Row] = {
      case Array(1, size) => Row(size)
    }
  }

  implicit object TwoD extends DimensionFactory[TwoD] {
    val createFunction: PartialFunction[Shape, TwoD] = {
      case Array(x, y) if x > 1 => TwoD(x,y)
    }
  }

  implicit object ThreeD extends DimensionFactory[ThreeD] {
    val createFunction: PartialFunction[Shape, ThreeD] = {
      case Array(x, y, z) => ThreeD(x,y,z)
    }
  }

  implicit object FourD extends DimensionFactory[FourD] {
    val createFunction: PartialFunction[Shape, FourD] = {
      case Array(x, y, z, f) => FourD(x,y,z,f)
    }
  }


}
