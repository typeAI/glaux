package glaux.linalg

import Dimension._
import Vol._
import glaux.statistics.{RealDistribution, Distribution}

trait Vol extends VolOperations {

  type Dimensionality <: Dimension
  def dimension: Dimensionality
  def sumAll: Double = seqView.sum
  def seqView: Seq[Double]
  def toArray: Array[Double] = seqView.toArray
}


trait VolOperations {

  def +(that: Vol): Vol

  def -(that: Vol): Vol

  /** element-by-element multiplication */
  def *(that: Vol): Vol

  /** matrix multiplication */
  def **(that: Vol): Vol

  /** matrix multiplication using Numpy syntax for arrays */
  def dot(that: Vol): Vol

  def /(that: Vol): Vol

  /** right division ... is this the correct symbol? */
  def \(that: Vol): Vol

  def +(that: Number): Vol

  def -(that: Number): Vol

  def *(that: Number): Vol

  def /(that: Number): Vol

  def \(that: Number): Vol

  def T: Vol

}

object Vol {
  
  type CanBuildFrom[From, V <: Vol] = From => V
  type VolBuilder[V <: Vol] = CanBuildFrom[(V#Dimensionality, Seq[Double]), V] //Seq instead of Iterable for performance concerns
  type GenVolBuilder[V <: Vol] = CanBuildFrom[(Dimension, Seq[Double]), V]
  type RowBuilder = VolBuilder[RowVector]
  type MatrixBuilder = VolBuilder[Matrix]
  type Vol3DBuilder = VolBuilder[Vol3D]


  implicit val toRow: CanBuildFrom[Vol, RowVector] = v => v.asInstanceOf[RowVector]
  implicit val toMatrix: CanBuildFrom[Vol, Matrix] = v => v.asInstanceOf[Matrix]

  implicit def toGen[V <: Vol](implicit gb: GenVolBuilder[V]): VolBuilder[V] = gb
  implicit class VolOps[V <: Vol : VolBuilder](self: V) {

    def map(f: Double => Double): V = (self.dimension, self.seqView.map(f))

    def fill(value: Double): V = map(_ => 0)

    def merge(v2: V)(f: (Double, Double) => Double) : V = {
      assert(self.dimension == v2.dimension)
      (self.dimension, self.seqView.zip(v2.seqView).map(f.tupled))
    }
  }

  def apply(dimension: Dimension, data: Seq[Double]) : Vol = (dimension, data)

}

trait RowVector extends Vol {
  type Dimensionality = Row
  def apply(index: Int) : Double
}

trait Matrix extends Vol {
  type Dimensionality = TwoD
}

trait Vol3D extends Vol {
  type Dimensionality = ThreeD
}


trait VolFactory[V <: Vol] {
  def apply(dimension: V#Dimensionality, data: Seq[Double])(implicit b: VolBuilder[V]) : V = b((dimension, data))

  def fill(dimension: V#Dimensionality, value: Double)(implicit b: VolBuilder[V]): V = apply(dimension, Array.fill(dimension.totalSize)(value))

  def sampleOf(dimension: V#Dimensionality, dist: RealDistribution, size: Int)(implicit b: VolBuilder[V]): Iterable[V] =
    1.until(size).map(_ => sampleOf(dimension, dist))

  def sampleOf(dimension: V#Dimensionality, dist: RealDistribution)(implicit b: VolBuilder[V]): V =
    apply(dimension, dist.sample(dimension.totalSize).toSeq)

}

object RowVector extends VolFactory[RowVector]{
  def apply(values: Double*): RowVector = apply(Dimension.Row(values.length), values)
}

object Vol3D extends VolFactory[Vol3D] {
  def apply(x: Int, y: Int, z: Int, data: Seq[Double]): Vol3D = apply(Dimension.ThreeD(x,y,z), data)
}

object Matrix extends VolFactory[Matrix]{
  def apply(x: Int, y: Int, data: Seq[Double]): Matrix = apply(Dimension.TwoD(x,y), data)
}
