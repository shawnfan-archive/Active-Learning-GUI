<template>
  <v-container>
    <v-row align="start" justify="center">
      <!-- Buttons and canvases -->
      <v-card class="pa-0 my-0">
        <v-container class="pa-0">
          <v-row class="pa-0 ma-0" align="center" justify="center">
            <!-- Buttons and Current Histogram -->
            <v-col cols="4">
              <v-row no-gutters class="ma-0 pa-0" align="stretch" justify="end">
                <v-col cols="12">
                  <v-row class="mb-12" no-gutters align="baseline" justify="end">
                    <!-- Undo -->
                    <v-col cols="12">
                      <v-row justify="end">
                        <v-btn class="ma-4" fab dark color="purple" v-on:click="undoCorrection">
                          <v-icon dark>mdi-undo</v-icon>
                        </v-btn>
                      </v-row>
                    </v-col>

                    <!-- Paintbrush -->
                    <v-col cols="12">
                      <v-row justify="end">
                        <v-menu left offset-y>
                          <template v-slot:activator="{ on }">
                            <v-btn class="ma-4" fab dark color="red" v-on="on">
                              <v-icon dark>mdi-pencil</v-icon>
                            </v-btn>
                          </template>

                          <v-col>
                            <v-btn-toggle>
                              <v-btn
                                v-for="parameter in button_parameters"
                                :key="parameter[0]"
                                fab
                                color="orange"
                                v-on:click="setTool('activate',parameter[0])"
                              >
                                <v-icon :size="parameter[1]">mdi-checkbox-blank-circle</v-icon>
                              </v-btn>
                            </v-btn-toggle>
                          </v-col>
                        </v-menu>
                      </v-row>
                    </v-col>

                    <!-- Eraser -->
                    <v-col cols="12">
                      <v-row justify="end">
                        <v-menu left offset-y>
                          <template v-slot:activator="{ on }">
                            <v-btn class="ma-4" fab dark color="cyan" v-on="on">
                              <v-icon dark>mdi-eraser</v-icon>
                            </v-btn>
                          </template>

                          <v-col>
                            <v-btn-toggle>
                              <v-btn
                                v-for="parameter in button_parameters"
                                :key="parameter[0]"
                                fab
                                color="blue"
                                v-on:click="setTool('deactivate',parameter[0])"
                              >
                                <v-icon :size="parameter[1]">mdi-checkbox-blank-circle</v-icon>
                              </v-btn>
                            </v-btn-toggle>
                          </v-col>

                          <v-col justify="center">
                            <v-btn text v-on:click="clearActivationMap()">
                              Clear All
                              <v-icon>mdi-close</v-icon>
                            </v-btn>
                          </v-col>
                        </v-menu>
                      </v-row>
                    </v-col>

                    <!-- Submit -->
                    <v-col cols="12">
                      <v-row justify="end">
                        <v-btn class="ma-4" fab dark color="success" v-on:click="submit">
                          <v-icon dark>mdi-upload</v-icon>
                        </v-btn>
                      </v-row>
                    </v-col>

                  </v-row>
                </v-col>
              </v-row>
            </v-col>
            <!-- Canvases -->
            <v-col cols="8">
              <v-row class="pa-0 ma-0" no-gutters align="center" justify="center">
                <v-col cols="12">
                  <v-header align="center"
                  >Image ID: {{ image.image_id }} | Disease: {{ image.disease }}</v-header>
                </v-col>

                <!--canvas for brain image-->
                <v-col cols="12">
                  <div class="canvas">
                    <canvas
                      id="image"
                      ref="image"
                      v-bind:width="canvas_width"
                      v-bind:height="canvas_height"
                    ></canvas>
                  </div>
                  <!-- canvas for cursor -->
                  <div class="canvas">
                    <canvas ref="cursor" v-bind:width="canvas_width" v-bind:height="canvas_height"></canvas>
                  </div>
                  <!--canvas for activation map-->
                  <div id="draw">
                    <canvas
                      ref="draw"
                      v-bind:width="canvas_width"
                      v-bind:height="canvas_height"
                      v-on:mousemove="correctActivation"
                    ></canvas>
                  </div>
                </v-col>
              </v-row>
            </v-col>
          </v-row>
        </v-container>
      </v-card>
    </v-row>
  </v-container>
</template>

<script>
export default {
  props: {
    image: Object,
    activation_map: Array,
    h: Number,
    w: Number,
    rescale: false,
    scale: Number
  },
  data() {
    return {
      canvas_width: this.w,
      canvas_height: this.h,
      graphics: {
        paintbrush: "rgb(254, 0, 0)",
        map_opacity: 0.8
      },
      button_parameters: [
        [20, 25],
        [10, 12.5],
        [5, 7.5]
      ],
      map: this.activation_map,
      tool: "deactivate",
      tool_started: false,
      tool_size: 10,
      paint: {
        points: [],
        tool: "",
        history: []
      }
    };
  },
  methods: {
    setTool: function(tool, tool_size) {
      this.tool = tool;
      this.tool_size = tool_size;
    },
    loadImage: function() {
      // //Load activation map
      // this.loadActivationMap();

      //Load new image onto canvas
      let canvas = this.$refs.image;
      let ctx = this.$refs.image.getContext("2d");
      //Construct canvas image from image.data
      let canvas_img = ctx.getImageData(0, 0, this.w, this.h);
      let compression_factor = this.computeRange(this.image.data) / 256;

      for (let i = 0; i < this.h; i += 1) {
        for (let j = 0; j < this.w; j += 1) {
          let pixel_index = 4 * (i * this.w + j);
          let pixel_value = Math.floor(
            this.image.data[i][j] / compression_factor
          );

          canvas_img.data[pixel_index] = pixel_value;
          canvas_img.data[pixel_index + 1] = canvas_img.data[pixel_index];
          canvas_img.data[pixel_index + 2] = canvas_img.data[pixel_index];
          canvas_img.data[pixel_index + 3] = 255;
        }
      }

      ctx.putImageData(canvas_img, 0, 0);

      // let imageObject = new Image()
      // imageObject.onload = () => {
      //     ctx.clearRect(0, 0, this.w, this.h);
      //     ctx.scale(10, 10);
      //     ctx.drawImage(imageObject, 0, 0)
      // }
      // imageObject.src=canvas.toDataURL();

      // Reinitialize history
      this.paint.history = [];
    },
    computeRange: function(nest_arr) {
      let minRow = nest_arr.map(function(row) {
        return Math.min.apply(Math, row);
      });
      let maxRow = nest_arr.map(function(row) {
        return Math.max.apply(Math, row);
      });

      return Math.max.apply(null, maxRow) - Math.min.apply(null, minRow);
    },
    computeMapThreshold: function(map) {
      let minRow = map.map(function(row) {
        return Math.min.apply(Math, row);
      });
      let map_min = Math.min.apply(null, minRow);

      return map_min + this.computeRange(map) / 2;
    },
    storeActivationMap: function() {
      //Emit corrections

      //Get RGBA array
      let map_ctx = this.$refs.draw.getContext("2d");
      let map_data = map_ctx.getImageData(0, 0, this.w, this.h);
      let corr_map = Array(this.h)
        .fill(0)
        .map(x => Array(this.w).fill(0));

      //Convert RGBA array into nested binary arrays
      for (
        let pixel_index = 0;
        pixel_index < map_data.data.length;
        pixel_index += 4
      ) {
        let row_index = Math.floor(pixel_index / 4 / this.w);
        let col_index = (pixel_index / 4) % this.w;
        if (map_data.data[pixel_index + 3] !== 0) {
          corr_map[row_index][col_index] = 1;
        } else {
          corr_map[row_index][col_index] = 0;
        }
      }
      console.log(this.image.image_id);
      this.$emit("update_activation_map", corr_map);
    },
    loadActivationMap: function() {
      console.log("hi");
      let canvas = this.$refs.draw;
      let ctx = this.$refs.draw.getContext("2d");
      ctx.strokeStyle = this.graphics.paintbrush;
      ctx.fillStyle = this.graphics.paintbrush;

      //Clear current activation map
      ctx.clearRect(0, 0, this.w, this.h);

      ctx.globalCompositeOperation = "source-over";

      let threshold = this.computeMapThreshold(this.activation_map);
      console.log(threshold);
      console.log(this.w);

      for (let row_index = 0; row_index < this.h; row_index++) {
        for (let column_index = 0; column_index < this.w; column_index++) {
          if (this.activation_map[row_index][column_index] > 0.5 * threshold) {
            ctx.fillRect(column_index, row_index, 1, 1);
          }
        }
      }
    },
    loadCursor: function(x, y) {
      let ctx = this.$refs.cursor.getContext("2d");
      ctx.clearRect(0, 0, this.canvas_width, this.canvas_height);
      ctx.fillStyle = this.graphics.paintbrush;
      ctx.beginPath();
      ctx.arc(x, y, this.tool_size / 2, 0, 2 * Math.PI);
      ctx.fill();
    },
    correctActivation: function(event) {
      // load cursor
      this.loadCursor(event.offsetX, event.offsetY);

      let canvas = this.$refs.draw;
      let ctx = this.$refs.draw.getContext("2d");

      //Set graphics parameters
      ctx.strokeStyle = this.graphics.paintbrush;
      ctx.fillStyle = this.graphics.paintbrush;
      ctx.lineJoin = ctx.lineCap = "round";
      ctx.lineWidth = this.tool_size;

      //Check if using eraser or painbrush
      if (this.tool === "deactivate") {
        ctx.globalCompositeOperation = "destination-out";
      } else {
        ctx.globalCompositeOperation = "source-over";
      }

      canvas.onmousedown = () => {
        this.paint.points = [];
        this.drawCircle(ctx, event.offsetX, event.offsetY);
        //Start drawing
        this.tool_started = true;
        this.paint.points.push({ x: event.offsetX, y: event.offsetY });
      };

      if (this.tool_started) {
        this.paint.points.push({ x: event.offsetX, y: event.offsetY });

        ctx.beginPath();

        let points_length = this.paint.points.length;

        //draw just the last segment
        ctx.moveTo(
          this.paint.points[points_length - 2].x,
          this.paint.points[points_length - 2].y
        );
        ctx.lineTo(
          this.paint.points[points_length - 1].x,
          this.paint.points[points_length - 1].y
        );

        ctx.stroke();
      }

      canvas.onmouseup = () => {
        //Stop drawing
        this.tool_started = !this.tool_started;
        //Draw circle
        this.drawCircle(ctx, event.offsetX, event.offsetY);
        this.paint.history.push(this.paint.points);
      };
    },
    undoCorrection: function() {
      this.paint.history.pop();

      let canvas = this.$refs.draw;
      let ctx = this.$refs.draw.getContext("2d");

      this.clearActivationMap();

      for (let i = 0; i < this.paint.history.length; i++) {
        let memory = this.paint.history[i];

        this.drawCircle(ctx, memory[0].x, memory[0].y);

        for (let j = 1; j < memory.length; j++) {
          ctx.beginPath();
          ctx.moveTo(memory[j - 1].x, memory[j - 1].y);
          ctx.lineTo(memory[j].x, memory[j].y);
          ctx.stroke();
        }

        this.drawCircle(
          ctx,
          memory[memory.length - 1].x,
          memory[memory.length - 1].y
        );
      }
    },
    clearActivationMap: function() {
      //Clear current activation map
      let ctx = this.$refs.draw.getContext("2d");
      ctx.clearRect(0, 0, this.w, this.h);
    },
    drawCircle: function(ctx, x, y) {
      ctx.beginPath();
      ctx.arc(x, y, this.tool_size / 2, 0, 2 * Math.PI);
      ctx.fill();
    },
    submit: function() {
      this.storeActivationMap();

      this.$emit("submit_corrections", true)
    }
  },
  crated() {},
  mounted() {
    this.loadActivationMap();
    this.loadImage();
  },
  beforeUpdate() {
    this.storeActivationMap();
  },
  updated() {
    this.loadActivationMap();
    this.loadImage();
  }
};
</script>

<style scoped>
.canvas {
  position: absolute;
  cursor: none;
}
#draw {
  cursor: none;
  opacity: 0.3;
}
</style>