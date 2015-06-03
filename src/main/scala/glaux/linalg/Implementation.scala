package glaux.linalg

import glaux.linalg.Vol._

trait Implementation {
  implicit val rBuilder: RowBuilder
  implicit val mBuilder: MatrixBuilder
  implicit val vBuilder: Vol3DBuilder
  implicit val genBuilder: GenVolBuilder[Vol]
}
