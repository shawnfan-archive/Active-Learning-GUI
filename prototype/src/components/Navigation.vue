<template>
  <nav>
    <v-app-bar dense flat app>
      <v-app-bar-nav-icon class="grey--text" @click="drawer = !drawer"></v-app-bar-nav-icon>

      <v-toolbar-title class="grey--text">Towards Active Learning</v-toolbar-title>

      <v-spacer></v-spacer>

      <v-toolbar-items>
        <v-btn text v-on:click="toggleSubmissionDialog">
          Save and Retrain Model
          <v-icon>mdi-upload</v-icon>
        </v-btn>
        <v-btn text>
          Settings
          <v-icon>mdi-settings</v-icon>
        </v-btn>
      </v-toolbar-items>
    </v-app-bar>

    <v-navigation-drawer v-model="drawer" absolute temporary class="primary">
      <v-list>
        <v-list-item v-for="link in links" :key="link.text" router :to="link.route">
          <v-list-item-icon>
            <v-icon class="white--text">{{ link.icon }}</v-icon>
          </v-list-item-icon>
          <v-list-item-content>
            <v-list-item-title class="white--text">{{ link.text }}</v-list-item-title>
          </v-list-item-content>
        </v-list-item>
      </v-list>
    </v-navigation-drawer>
  </nav>
</template>

<script>
export default {
  data() {
    return {
      drawer: false,
      links: [
        {
          icon: "mdi-brain",
          text: "Stroke Segmentation",
          route: "/"
        },
        {
          icon: "mdi-drawing",
          text: "Playground",
          route: "/playground"
        }
      ]
    };
  }
};
</script>