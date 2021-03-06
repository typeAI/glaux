package glaux.linearalgebra

import glaux.linearalgebra.Dimension.{Row, ThreeD, TwoD}
import org.specs2.mutable.Specification

class TensorSpec
  extends Specification {

  "Normal Vols" >> {
    val dim = ThreeD(1, 1, 3)

    "* op keeps dimension" >> {
      val vol = Tensor(dim, Seq(2, 3, 4))
      val vol2: Tensor = vol * vol
      vol2 must_== Tensor(dim, Seq(4, 9, 16))
    }

    "* scalar keeps dimension" >> {
      val vol = Tensor(dim, Seq(2, 3, 4))
      val vol2: Tensor = vol * 0.5
      vol2 must_== Tensor(dim, Seq(1, 1.5, 2))
    }

    "sum correctly" >> {
      val m = Matrix(2, 2, Seq(1, 2, 3, 4))
      m.sumAll must_== 10
    }

    "map" >> {
      "with immutability" >> {
        val m = Matrix(2, 2, Seq(1, 2, 3, 4))
        m.map(_ * 2)
        m must_== Matrix(2, 2, Seq(1, 2, 3, 4))
      }

      "correctly" >> {
        val m = Matrix(2, 2, Seq(1, 2, 3, 4))
        val result = m.map(_ * 2)
        result must_== Matrix(2, 2, Seq(2, 4, 6, 8))
      }
    }

    "toRow" >> {
      "matrix" >> {
        val m = Matrix(2, 2, Seq(1, 2, 3, 4))
        m.toRowVector must_== RowVector(1, 2, 3, 4)
      }
      "row vector" >> {
        val r = RowVector(1, 2, 3, 4)
        r.toRowVector must_== RowVector(1, 2, 3, 4)
      }

    }

    "merge" >> {

      "correctly" >> {
        val m = Vol(ThreeD(2, 2, 2), 1.until(9).map(_.toDouble))
        val m2 = Vol(ThreeD(2, 2, 2), 9.until(1, -1).map(_.toDouble))
        val result = m.merge(m2)(_ + _)
        result must_== Vol(ThreeD(2, 2, 2), Seq.fill(8)(10d))
      }
    }

    "transpose" >> {
      RowVector(0, 1, 0, 0).T must_== Matrix(4, 1, Seq(0, 1, 0, 0))
    }

    "dot" >> {
      val r1 = RowVector(3, 3, 1, 2)
      val r2 = RowVector(0, 1, 0, 2)
      (r1 dot r2) === 7
    }

    "uniform" >> {
      val m = Vol.fill(ThreeD(2, 2, 2), 10)
      m must_== Vol(2, 2, 2, Seq.fill(8)(10d))

    }

    "normal" >> {
      import glaux.statistics.distributions.normal
      val data = RowVector.sampleOf(Row(3), normal(10, 3), 100).toSeq.flatMap(_.seqView)
      val avg = data.sum / data.size
      val devs = data.map(value ⇒ (value - avg) * (value - avg))
      val std = Math.sqrt(devs.sum / data.size)
      avg must beCloseTo(10.0 within 2.significantFigures)
      std must beCloseTo(3.0 within 1.significantFigures)
    }

  }

  "Generic VolOps " >> {
    "map correctly" >> {
      val d: TwoD = TwoD(3, 1)
      val m: Tensor = Matrix.fill(d, 0.5)
      val result = m.map(_ * 2)
      result must_== Matrix.fill(d, 1)
    }
  }

  "RowVector Vols" >> {
    "sum works correctly" >> {
      RowVector(1, 3, 4, 5).sumAll === 13
    }
    "* works correctly" >> {
      val result: Tensor = RowVector(1, 3, 4, 5) * 3
      result must_== (RowVector(3, 9, 12, 15))
    }

  }

}
