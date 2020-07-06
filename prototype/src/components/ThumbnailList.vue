<template>
  <v-container>
    <v-row justify="center">
      <v-card>
        <v-list class="overflow-y-auto" max-height="550">
          <v-list-item-group>
            <v-list-item class="ma-4" v-for="image in images" :key="image.id">
              <v-card class="pa-2" width="100" height="105" raised>
                <v-img
                  class="my-0"
                  contain
                  :src="require(`../assets/${image.path}`)"
                  weight="75"
                  height="75"
                  v-on:click="switchImage(image)"
                ></v-img>
                <v-card-subtitle class="pa-0">{{image.image_id}}</v-card-subtitle>
              </v-card>
            </v-list-item>
          </v-list-item-group>
        </v-list>
      </v-card>
    </v-row>
  </v-container>
</template>

<script>
export default {
    props: {
        images: Array,
    },
    methods: {
        switchImage: function(image) {
            this.$emit("switch_image", image)
        }
    }
}
</script>

<style scoped>
</style>